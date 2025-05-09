import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import pulp as pl
import json
import uuid
import datetime

def calculate_distance(origin_idx, origin_position, dest_position, origins_df):
    """
    Calcula a distância entre os centros de massa em metros, 
    adicionando DT fixo para empréstimos
    """
    # Distância básica entre os centros de massa
    base_distance = abs(dest_position - origin_position)
    
    # Verifica se a origem é um empréstimo e tem DT fixo definido
    if 'DT Fixo (m)' in origins_df.columns:
        if pd.notna(origins_df.loc[origin_idx, 'DT Fixo (m)']):
            # Adiciona o DT fixo à distância básica
            dt_fixo = origins_df.loc[origin_idx, 'DT Fixo (m)']
            return base_distance + dt_fixo
    
    return base_distance

def check_feasibility(origins_df, destinations_df):
    """
    Verifica se o problema tem uma solução factível
    """
    total_volume_origem = origins_df['Volume disponível (m³)'].sum()
    total_volume_destino = destinations_df['Volume CFT (m³)'].fillna(0).sum() + destinations_df['Volume CA (m³)'].fillna(0).sum()
    
    if total_volume_origem < total_volume_destino:
        return False, f"Volume total de origem ({total_volume_origem:.2f} m³) é menor que o volume total de destino ({total_volume_destino:.2f} m³)."
    
    # Verifica se há material suficiente com ISC adequado para CFT
    for d_idx, d_row in destinations_df.iterrows():
        if pd.isna(d_row['Volume CFT (m³)']) or d_row['Volume CFT (m³)'] <= 0:
            continue
            
        isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
        valid_origins = origins_df[origins_df['ISC'] >= isc_min]
        
        if valid_origins.empty:
            return False, f"Não há origens com ISC suficiente para o destino {d_idx} (ISC min: {isc_min})."
        
        total_valid_volume = valid_origins['Volume disponível (m³)'].sum()
        if total_valid_volume < d_row['Volume CFT (m³)']:
            return False, f"Volume disponível com ISC adequado ({total_valid_volume:.2f} m³) é menor que o necessário para CFT no destino {d_idx} ({d_row['Volume CFT (m³)']:.2f} m³)."
    
    return True, "O problema parece ter uma solução factível."

def identify_emprestimo_types(origins_df):
    """
    Identifica os tipos de empréstimos (laterais ou concentrados) baseado no tipo
    """
    emprestimos_laterais_idx = origins_df[
        origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True)
    ].index.tolist()
    
    emprestimos_concentrados_idx = origins_df[
        (origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)) & 
        (~origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
    ].index.tolist()
    
    cortes_idx = [idx for idx in origins_df.index 
                 if idx not in emprestimos_laterais_idx and idx not in emprestimos_concentrados_idx]
    
    return cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx

def optimize_distribution(origins_df, destinations_df, time_limit=1800, favor_cortes=False, 
                          max_dist_cortes=None, max_dist_emprestimos=None, fixed_allocations=None):
    """
    Otimiza a distribuição de materiais usando programação linear
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
        time_limit: Tempo limite para o solver (em segundos)
        favor_cortes: Se True, favorece materiais de corte sobre empréstimos
        max_dist_cortes: Distância máxima para transporte de material de corte (em metros)
        max_dist_emprestimos: Distância máxima para transporte de material de empréstimos (em metros)
        fixed_allocations: Lista de dicionários com alocações fixas [{'origem': 'id_origem', 
                          'destino': 'id_destino', 'volume': valor, 'tipo': 'CFT|CA'}]
    """
    # Cria o problema de otimização
    problem = pl.LpProblem("Terraplenagem_Otimizacao", pl.LpMinimize)
    
    # Cópias dos volumes para trabalho
    available_volumes = origins_df['Volume disponível (m³)'].copy()
    needed_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    needed_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    # Processa alocações fixas, se fornecidas
    fixed_volumes_origin = {}  # Para rastrear volumes já alocados fixos por origem
    fixed_volumes_dest_cft = {}  # Para destinos CFT
    fixed_volumes_dest_ca = {}  # Para destinos CA
    
    if fixed_allocations:
        print("Aplicando alocações fixas:")
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            print(f"  Alocação fixa: {o_idx} → {d_idx}, {volume} m³ ({tipo})")
            
            # Registra o volume fixo
            if o_idx not in fixed_volumes_origin:
                fixed_volumes_origin[o_idx] = 0
            fixed_volumes_origin[o_idx] += volume
            
            # Registra por tipo
            if tipo == 'CFT':
                if d_idx not in fixed_volumes_dest_cft:
                    fixed_volumes_dest_cft[d_idx] = 0
                fixed_volumes_dest_cft[d_idx] += volume
                
                # Reduz o volume necessário
                needed_cft[d_idx] -= volume
            elif tipo == 'CA':
                if d_idx not in fixed_volumes_dest_ca:
                    fixed_volumes_dest_ca[d_idx] = 0
                fixed_volumes_dest_ca[d_idx] += volume
                
                # Reduz o volume necessário
                needed_ca[d_idx] -= volume
            
            # Reduz o volume disponível
            available_volumes[o_idx] -= volume
    
    # Identificar origens que são empréstimos
    emprestimos_idx = origins_df[
        origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)
    ].index.tolist()
    
    # Identificar origens que são cortes (não-empréstimos)
    cortes_idx = [idx for idx in origins_df.index if idx not in emprestimos_idx]
    
    print(f"Empréstimos identificados: {emprestimos_idx}")
    print(f"Cortes identificados: {cortes_idx}")
    
    # Calcula matriz de distâncias considerando DT fixo para empréstimos
    distances = {}
    adjusted_distances = {}  # Distâncias ajustadas considerando distâncias máximas
    
    for o_idx, o_row in origins_df.iterrows():
        for d_idx, d_row in destinations_df.iterrows():
            # Calcula distância básica
            dist = calculate_distance(
                o_idx,
                o_row['Centro de Massa (m)'], 
                d_row['Centro de Massa (m)'],
                origins_df
            )
            distances[(o_idx, d_idx)] = dist
            
            # Por padrão, usa a distância calculada
            adjusted_dist = dist
            
            # Aplica penalização para cortes que excedem a distância máxima
            if max_dist_cortes is not None and o_idx in cortes_idx and dist > max_dist_cortes:
                adjusted_dist = dist * 2.0  # Penalização para cortes
                print(f"Distância ajustada para corte {o_idx}->{d_idx}: {dist} -> {adjusted_dist} (excede máximo de {max_dist_cortes}m)")
            
            # Aplica penalização para empréstimos que excedem a distância máxima
            if max_dist_emprestimos is not None and o_idx in emprestimos_idx and dist > max_dist_emprestimos:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos
                print(f"Distância ajustada para empréstimo {o_idx}->{d_idx}: {dist} -> {adjusted_dist} (excede máximo de {max_dist_emprestimos}m)")
            
            adjusted_distances[(o_idx, d_idx)] = adjusted_dist
    
    # Normaliza as distâncias ajustadas para evitar problemas numéricos
    max_adj_distance = max(adjusted_distances.values()) if adjusted_distances else 1
    for key in adjusted_distances:
        adjusted_distances[key] /= max_adj_distance  # Normaliza para [0,1]
    
    # Verifica se há alguma configuração de ISC impossível
    feasible = True
    for d_idx, d_row in destinations_df.iterrows():
        if needed_cft[d_idx] > 0:
            isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
            valid_origins = origins_df[origins_df['ISC'] >= isc_min]
            
            if valid_origins.empty:
                st.warning(f"Não há origem com ISC adequado para o destino {d_idx} (ISC min: {isc_min})")
                feasible = False
    
    if not feasible:
        st.error("Problema infactível devido a restrições de ISC incompatíveis.")
        return None  # Retorna None para indicar que o problema é infactível
    
    # Cria variáveis de decisão para CFT
    cft_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_cft[d_idx] > 0:
                # Verifica compatibilidade de ISC
                isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido']
                if pd.isna(isc_min) or origins_df.loc[o_idx, 'ISC'] >= isc_min:
                    cft_vars[(o_idx, d_idx)] = pl.LpVariable(
                        f"CFT_{o_idx}_{d_idx}", 
                        lowBound=0
                    )
    
    # Cria variáveis de decisão para CA
    ca_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_ca[d_idx] > 0:
                ca_vars[(o_idx, d_idx)] = pl.LpVariable(
                    f"CA_{o_idx}_{d_idx}", 
                    lowBound=0
                )
    
    # Cria variáveis para bota-fora, mas somente para origens que não são empréstimos
    bota_fora_vars = {}
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx:  # Somente origens que não são empréstimos
            bota_fora_vars[o_idx] = pl.LpVariable(f"BF_{o_idx}", lowBound=0)
    
    # Cria variáveis de folga para destinos (para identificar onde não está atendendo)
    cft_slack_vars = {}
    ca_slack_vars = {}
    
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            cft_slack_vars[d_idx] = pl.LpVariable(f"CFT_SLACK_{d_idx}", lowBound=0)
        if needed_ca[d_idx] > 0:
            ca_slack_vars[d_idx] = pl.LpVariable(f"CA_SLACK_{d_idx}", lowBound=0)
    
    # Variáveis para volume não utilizado de cada empréstimo
    emprestimo_nao_utilizado = {}
    for o_idx in emprestimos_idx:
        emprestimo_nao_utilizado[o_idx] = pl.LpVariable(f"NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    # Função objetivo depende da escolha de favorecimento
    if favor_cortes:
        # Função objetivo que favorece materiais de corte
        problem += (
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]) +
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]) +
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]) +
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()]) +
            0.1 * pl.lpSum([bota_fora_vars[o_idx] for o_idx in bota_fora_vars.keys()])  # Penalização para bota-fora
        )
    else:
        # Função objetivo que prioriza apenas menor distância
        problem += (
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]) +
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]) +
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]) +
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()])
        )
    
    # Restrição: volume total distribuído de cada origem + bota-fora = volume disponível 
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx:  # Origens que não são empréstimos
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
                bota_fora_vars[o_idx] == available_volumes[o_idx],
                f"Conservacao_Volume_{o_idx}"
            )
        else:  # Para empréstimos
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
                emprestimo_nao_utilizado[o_idx] == available_volumes[o_idx],
                f"Conservacao_Volume_{o_idx}"
            )
    
    # Restrição: volume recebido de CFT em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in cft_vars]) + 
                cft_slack_vars[d_idx] == needed_cft[d_idx],
                f"Atendimento_CFT_{d_idx}"
            )
    
    # Restrição: volume recebido de CA em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_ca[d_idx] > 0:
            problem += (
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in ca_vars]) + 
                ca_slack_vars[d_idx] == needed_ca[d_idx],
                f"Atendimento_CA_{d_idx}"
            )
    
    # Configurações do solver
    if len(origins_df) * len(destinations_df) > 10000:  # Problemas muito grandes
        st.warning("Problema muito grande detectado. Ajustando parâmetros do solver...")
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            gapRel=0.05,
            options=['presolve on', 'strong branching on', 'gomory on']
        )
    else:
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit,
            gapRel=0.01,
            options=['presolve on']
        )
    
    # Resolve o problema
    print("Iniciando otimização...")
    status = problem.solve(solver)
    print("Otimização concluída!")
    
    # Verifica o status da solução
    status_text = pl.LpStatus[status]
    print(f"Status da otimização: {status_text}")
    print(f"Valor da função objetivo: {pl.value(problem.objective)}")
    
    # Extrai os resultados
    cft_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    ca_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    bota_fora = pd.Series(0, index=origins_df.index)
    
    # Adiciona primeiro as alocações fixas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            if tipo == 'CFT':
                cft_distribution.loc[o_idx, d_idx] = volume
            elif tipo == 'CA':
                ca_distribution.loc[o_idx, d_idx] = volume
    
    # Adiciona as alocações calculadas pelo otimizador
    for (o_idx, d_idx), var in cft_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            cft_distribution.loc[o_idx, d_idx] += var.value()
    
    for (o_idx, d_idx), var in ca_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            ca_distribution.loc[o_idx, d_idx] += var.value()
    
    # Somente origens não-empréstimo vão para bota-fora
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx and o_idx in bota_fora_vars:
            var = bota_fora_vars[o_idx]
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora[o_idx] = var.value()
    
    # Verifica valores de folga (não atendimento)
    cft_slack = {}
    ca_slack = {}
    for d_idx, var in cft_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            cft_slack[d_idx] = var.value()
            print(f"CFT não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    for d_idx, var in ca_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            ca_slack[d_idx] = var.value()
            print(f"CA não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    # Volume não utilizado de empréstimos
    emprestimos_nao_utilizados = {}
    for o_idx, var in emprestimo_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo {o_idx}: {var.value():.2f} m³")
    
    # Se houver problemas sérios com o não atendimento, retorna None
    if (sum(cft_slack.values()) > 0.05 * destinations_df['Volume CFT (m³)'].fillna(0).sum() or 
        sum(ca_slack.values()) > 0.05 * destinations_df['Volume CA (m³)'].fillna(0).sum()):
        print("Não atendimento significativo.")
        st.warning("O solver não conseguiu atender a um percentual significativo dos volumes necessários.")
    
    # Calcula volumes não atendidos
    # Considerando os volumes originais (incluindo os fixados)
    original_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    original_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    distributed_cft = cft_distribution.sum(axis=0)
    distributed_ca = ca_distribution.sum(axis=0)
    
    remaining_cft = original_cft - distributed_cft
    remaining_ca = original_ca - distributed_ca
    
    # Substituindo valores negativos pequenos por zero
    remaining_cft = remaining_cft.map(lambda x: max(0, x))
    remaining_ca = remaining_ca.map(lambda x: max(0, x))
    
    # Desnormaliza as distâncias originais para cálculos de resultado
    distances_df = pd.DataFrame(index=origins_df.index, columns=destinations_df.index)
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            distances_df.loc[o_idx, d_idx] = distances[(o_idx, d_idx)]
    
    # Calcular momento de transporte (m³·m) usando distâncias REAIS
    momento_cft = sum(cft_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                     for o_idx in origins_df.index
                     for d_idx in destinations_df.index
                     if cft_distribution.loc[o_idx, d_idx] > 0)
    
    momento_ca = sum(ca_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                    for o_idx in origins_df.index
                    for d_idx in destinations_df.index
                    if ca_distribution.loc[o_idx, d_idx] > 0)
    
    momento_total = momento_cft + momento_ca
    
    # Distância média de transporte (m)
    volume_total_distribuido = cft_distribution.sum().sum() + ca_distribution.sum().sum()
    if volume_total_distribuido > 0:
        dmt = momento_total / volume_total_distribuido
    else:
        dmt = 0
    
    # Volume total não utilizado de empréstimos
    total_emprestimo_nao_utilizado = sum(emprestimos_nao_utilizados.values())
    
    return {
        'cft': cft_distribution,
        'ca': ca_distribution,
        'bota_fora': bota_fora,
        'distances': distances_df,
        'remaining_cft': remaining_cft,
        'remaining_ca': remaining_ca,
        'momento_total': momento_total,
        'dmt': dmt,
        'status': status_text,
        'emprestimos_nao_utilizados': emprestimos_nao_utilizados,
        'total_emprestimo_nao_utilizado': total_emprestimo_nao_utilizado,
        'favor_cortes': favor_cortes,
        'max_dist_cortes': max_dist_cortes,
        'max_dist_emprestimos': max_dist_emprestimos,
        'fixed_allocations': fixed_allocations if fixed_allocations else []
    }

def optimize_distribution_advanced(origins_df, destinations_df, time_limit=1800, favor_cortes=False, 
                                  max_dist_cortes=None, max_dist_emprestimos_laterais=None, 
                                  max_dist_emprestimos_concentrados=None, fixed_allocations=None,
                                  cortes_idx=None, emprestimos_laterais_idx=None, emprestimos_concentrados_idx=None):
    """
    Versão avançada da função optimize_distribution que considera diferentes tipos de empréstimos
    """
    # Cria o problema de otimização
    problem = pl.LpProblem("Terraplenagem_Otimizacao", pl.LpMinimize)
    
    # Cópias dos volumes para trabalho
    available_volumes = origins_df['Volume disponível (m³)'].copy()
    needed_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    needed_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    # Processa alocações fixas, se fornecidas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            # Reduz volumes conforme alocações fixas
            if tipo == 'CFT':
                needed_cft[d_idx] -= volume
            elif tipo == 'CA':
                needed_ca[d_idx] -= volume
            
            # Reduz o volume disponível
            available_volumes[o_idx] -= volume
    
    # Se não foram fornecidos, identifica os tipos de origens
    if cortes_idx is None or emprestimos_laterais_idx is None or emprestimos_concentrados_idx is None:
        cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx = identify_emprestimo_types(origins_df)
    
    # União de todos os tipos de empréstimos para regras gerais
    emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
    
    # Calcula matriz de distâncias considerando DT fixo para empréstimos
    distances = {}
    adjusted_distances = {}  # Distâncias ajustadas considerando distâncias máximas
    
    for o_idx, o_row in origins_df.iterrows():
        for d_idx, d_row in destinations_df.iterrows():
            # Calcula distância básica
            dist = calculate_distance(
                o_idx,
                o_row['Centro de Massa (m)'], 
                d_row['Centro de Massa (m)'],
                origins_df
            )
            distances[(o_idx, d_idx)] = dist
            
            # Por padrão, usa a distância calculada
            adjusted_dist = dist
            
            # Aplica penalização para cortes que excedem a distância máxima
            if max_dist_cortes is not None and o_idx in cortes_idx and dist > max_dist_cortes:
                adjusted_dist = dist * 2.0  # Penalização para cortes
            
            # Aplica penalização para empréstimos laterais que excedem a distância máxima
            if max_dist_emprestimos_laterais is not None and o_idx in emprestimos_laterais_idx and dist > max_dist_emprestimos_laterais:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos laterais
            
            # Aplica penalização para empréstimos concentrados que excedem a distância máxima
            if max_dist_emprestimos_concentrados is not None and o_idx in emprestimos_concentrados_idx and dist > max_dist_emprestimos_concentrados:
                adjusted_dist = dist * 2.0  # Penalização para empréstimos concentrados
            
            adjusted_distances[(o_idx, d_idx)] = adjusted_dist
    
    # Normaliza as distâncias ajustadas para evitar problemas numéricos
    max_adj_distance = max(adjusted_distances.values()) if adjusted_distances else 1
    for key in adjusted_distances:
        adjusted_distances[key] /= max_adj_distance  # Normaliza para [0,1]
    
    # Verifica se há alguma configuração de ISC impossível
    feasible = True
    for d_idx, d_row in destinations_df.iterrows():
        if needed_cft[d_idx] > 0:
            isc_min = d_row['ISC mínimo exigido'] if pd.notna(d_row['ISC mínimo exigido']) else 0
            valid_origins = origins_df[origins_df['ISC'] >= isc_min]
            
            if valid_origins.empty:
                feasible = False
    
    if not feasible:
        return None  # Retorna None para indicar que o problema é infactível
    
    # Cria variáveis de decisão para CFT
    cft_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_cft[d_idx] > 0:
                # Verifica compatibilidade de ISC
                isc_min = destinations_df.loc[d_idx, 'ISC mínimo exigido'] if pd.notna(destinations_df.loc[d_idx, 'ISC mínimo exigido']) else 0
                if pd.isna(isc_min) or origins_df.loc[o_idx, 'ISC'] >= isc_min:
                    cft_vars[(o_idx, d_idx)] = pl.LpVariable(
                        f"CFT_{o_idx}_{d_idx}", 
                        lowBound=0
                    )
    
    # Cria variáveis de decisão para CA
    ca_vars = {}
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            if needed_ca[d_idx] > 0:
                ca_vars[(o_idx, d_idx)] = pl.LpVariable(
                    f"CA_{o_idx}_{d_idx}", 
                    lowBound=0
                )
    
    # Cria variáveis para bota-fora, mas somente para origens que não são empréstimos
    bota_fora_vars = {}
    for o_idx in origins_df.index:
        if o_idx not in emprestimos_idx:  # Somente origens que não são empréstimos
            bota_fora_vars[o_idx] = pl.LpVariable(f"BF_{o_idx}", lowBound=0)
    
    # Cria variáveis de folga para destinos (para identificar onde não está atendendo)
    cft_slack_vars = {}
    ca_slack_vars = {}
    
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            cft_slack_vars[d_idx] = pl.LpVariable(f"CFT_SLACK_{d_idx}", lowBound=0)
        if needed_ca[d_idx] > 0:
            ca_slack_vars[d_idx] = pl.LpVariable(f"CA_SLACK_{d_idx}", lowBound=0)
    
    # Variáveis para volume não utilizado de cada empréstimo, separado por tipo
    emprestimo_lateral_nao_utilizado = {}
    for o_idx in emprestimos_laterais_idx:
        emprestimo_lateral_nao_utilizado[o_idx] = pl.LpVariable(f"LATERAL_NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    emprestimo_concentrado_nao_utilizado = {}
    for o_idx in emprestimos_concentrados_idx:
        emprestimo_concentrado_nao_utilizado[o_idx] = pl.LpVariable(f"CONCENTRADO_NAO_UTILIZADO_{o_idx}", lowBound=0)
    
    # Função objetivo depende da escolha de favorecimento
    if favor_cortes:
        # Função objetivo que favorece materiais de corte
        problem += (
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]) +
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]) +
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]) +
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()]) +
            0.1 * pl.lpSum([bota_fora_vars[o_idx] for o_idx in bota_fora_vars.keys()])  # Penalização para bota-fora
        )
    else:
        # Função objetivo que prioriza apenas menor distância
        problem += (
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * cft_vars[(o_idx, d_idx)] for (o_idx, d_idx) in cft_vars.keys()]) +
            pl.lpSum([adjusted_distances[(o_idx, d_idx)] * ca_vars[(o_idx, d_idx)] for (o_idx, d_idx) in ca_vars.keys()]) +
            100 * pl.lpSum([cft_slack_vars[d_idx] for d_idx in cft_slack_vars.keys()]) +
            100 * pl.lpSum([ca_slack_vars[d_idx] for d_idx in ca_slack_vars.keys()])
        )
    
    # Restrição: volume total distribuído de cada origem + bota-fora = volume disponível 
    for o_idx in origins_df.index:
        if o_idx in cortes_idx:  # Para cortes
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
                bota_fora_vars[o_idx] == available_volumes[o_idx],
                f"Conservacao_Volume_Corte_{o_idx}"
            )
        elif o_idx in emprestimos_laterais_idx:  # Para empréstimos laterais
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
                emprestimo_lateral_nao_utilizado[o_idx] == available_volumes[o_idx],
                f"Conservacao_Volume_Emp_Lateral_{o_idx}"
            )
        elif o_idx in emprestimos_concentrados_idx:  # Para empréstimos concentrados
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in cft_vars]) +
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for d_idx in destinations_df.index if (o_idx, d_idx) in ca_vars]) +
                emprestimo_concentrado_nao_utilizado[o_idx] == available_volumes[o_idx],
                f"Conservacao_Volume_Emp_Concentrado_{o_idx}"
            )
    
    # Restrição: volume recebido de CFT em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_cft[d_idx] > 0:
            problem += (
                pl.lpSum([cft_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in cft_vars]) + 
                cft_slack_vars[d_idx] == needed_cft[d_idx],
                f"Atendimento_CFT_{d_idx}"
            )
    
    # Restrição: volume recebido de CA em cada destino + folga = volume necessário
    for d_idx in destinations_df.index:
        if needed_ca[d_idx] > 0:
            problem += (
                pl.lpSum([ca_vars.get((o_idx, d_idx), 0) for o_idx in origins_df.index if (o_idx, d_idx) in ca_vars]) + 
                ca_slack_vars[d_idx] == needed_ca[d_idx],
                f"Atendimento_CA_{d_idx}"
            )
    
    # Configurações do solver
    if len(origins_df) * len(destinations_df) > 10000:  # Problemas muito grandes
        st.warning("Problema muito grande detectado. Ajustando parâmetros do solver...")
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            gapRel=0.05,
            options=['presolve on', 'strong branching on', 'gomory on']
        )
    else:
        solver = pl.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit,
            gapRel=0.01,
            options=['presolve on']
        )
    
    # Resolve o problema
    print("Iniciando otimização avançada...")
    status = problem.solve(solver)
    print("Otimização concluída!")
    
    # Verifica o status da solução
    status_text = pl.LpStatus[status]
    print(f"Status da otimização: {status_text}")
    print(f"Valor da função objetivo: {pl.value(problem.objective)}")
    
    # Extrai os resultados
    cft_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    ca_distribution = pd.DataFrame(0, index=origins_df.index, columns=destinations_df.index)
    bota_fora = pd.Series(0, index=origins_df.index)
    
    # Adiciona primeiro as alocações fixas
    if fixed_allocations:
        for alloc in fixed_allocations:
            o_idx = alloc['origem']
            d_idx = alloc['destino']
            volume = float(alloc['volume'])
            tipo = alloc['tipo']
            
            if tipo == 'CFT':
                cft_distribution.loc[o_idx, d_idx] = volume
            elif tipo == 'CA':
                ca_distribution.loc[o_idx, d_idx] = volume
    
    # Adiciona as alocações calculadas pelo otimizador
    for (o_idx, d_idx), var in cft_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            cft_distribution.loc[o_idx, d_idx] += var.value()
    
    for (o_idx, d_idx), var in ca_vars.items():
        if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
            ca_distribution.loc[o_idx, d_idx] += var.value()
    
    # Somente origens não-empréstimo vão para bota-fora
    for o_idx in cortes_idx:
        if o_idx in bota_fora_vars:
            var = bota_fora_vars[o_idx]
            if var.value() is not None and var.value() > 1e-6:  # Tolerância numérica
                bota_fora[o_idx] = var.value()
    
    # Verifica valores de folga (não atendimento)
    cft_slack = {}
    ca_slack = {}
    for d_idx, var in cft_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            cft_slack[d_idx] = var.value()
            print(f"CFT não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    for d_idx, var in ca_slack_vars.items():
        if var.value() is not None and var.value() > 1e-6:
            ca_slack[d_idx] = var.value()
            print(f"CA não atendido no destino {d_idx}: {var.value():.2f} m³")
    
    # Volume não utilizado de empréstimos separados por tipo
    emprestimos_laterais_nao_utilizados = {}
    for o_idx, var in emprestimo_lateral_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_laterais_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo lateral {o_idx}: {var.value():.2f} m³")
    
    emprestimos_concentrados_nao_utilizados = {}
    for o_idx, var in emprestimo_concentrado_nao_utilizado.items():
        if var.value() is not None:
            emprestimos_concentrados_nao_utilizados[o_idx] = var.value()
            print(f"Volume não utilizado do empréstimo concentrado {o_idx}: {var.value():.2f} m³")
    
    # Unindo todos os empréstimos não utilizados para análise
    emprestimos_nao_utilizados = {**emprestimos_laterais_nao_utilizados, **emprestimos_concentrados_nao_utilizados}
    
    # Se houver problemas sérios com o não atendimento, retorna aviso
    if (sum(cft_slack.values()) > 0.05 * destinations_df['Volume CFT (m³)'].fillna(0).sum() or 
        sum(ca_slack.values()) > 0.05 * destinations_df['Volume CA (m³)'].fillna(0).sum()):
        print("Não atendimento significativo.")
        st.warning("O solver não conseguiu atender a um percentual significativo dos volumes necessários.")
    
    # Calcula volumes não atendidos
    # Considerando os volumes originais (incluindo os fixados)
    original_cft = destinations_df['Volume CFT (m³)'].copy().fillna(0)
    original_ca = destinations_df['Volume CA (m³)'].copy().fillna(0)
    
    distributed_cft = cft_distribution.sum(axis=0)
    distributed_ca = ca_distribution.sum(axis=0)
    
    remaining_cft = original_cft - distributed_cft
    remaining_ca = original_ca - distributed_ca
    
    # Substituindo valores negativos pequenos por zero
    remaining_cft = remaining_cft.map(lambda x: max(0, x))
    remaining_ca = remaining_ca.map(lambda x: max(0, x))
    
    # Desnormaliza as distâncias originais para cálculos de resultado
    distances_df = pd.DataFrame(index=origins_df.index, columns=destinations_df.index)
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            distances_df.loc[o_idx, d_idx] = distances[(o_idx, d_idx)]
    
    # Calcular momento de transporte (m³·m) usando distâncias REAIS
    momento_cft = sum(cft_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                     for o_idx in origins_df.index
                     for d_idx in destinations_df.index
                     if cft_distribution.loc[o_idx, d_idx] > 0)
    
    momento_ca = sum(ca_distribution.loc[o_idx, d_idx] * distances_df.loc[o_idx, d_idx]
                    for o_idx in origins_df.index
                    for d_idx in destinations_df.index
                    if ca_distribution.loc[o_idx, d_idx] > 0)
    
    momento_total = momento_cft + momento_ca
    
    # Distância média de transporte (m)
    volume_total_distribuido = cft_distribution.sum().sum() + ca_distribution.sum().sum()
    if volume_total_distribuido > 0:
        dmt = momento_total / volume_total_distribuido
    else:
        dmt = 0
    
    # Volumes totais não utilizados por tipo de empréstimo
    total_emprestimo_lateral_nao_utilizado = sum(emprestimos_laterais_nao_utilizados.values())
    total_emprestimo_concentrado_nao_utilizado = sum(emprestimos_concentrados_nao_utilizados.values())
    total_emprestimo_nao_utilizado = total_emprestimo_lateral_nao_utilizado + total_emprestimo_concentrado_nao_utilizado
    
    # Análise por tipo de origem
    volume_por_tipo = {
        'cortes': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                           for idx in cortes_idx),
            'bota_fora': sum(bota_fora[idx] for idx in cortes_idx),
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in cortes_idx)
        },
        'emprestimos_laterais': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                            for idx in emprestimos_laterais_idx),
            'nao_utilizado': total_emprestimo_lateral_nao_utilizado,
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in emprestimos_laterais_idx)
        },
        'emprestimos_concentrados': {
            'utilizado': sum(cft_distribution.loc[idx].sum() + ca_distribution.loc[idx].sum() 
                             for idx in emprestimos_concentrados_idx),
            'nao_utilizado': total_emprestimo_concentrado_nao_utilizado,
            'total': sum(origins_df.loc[idx, 'Volume disponível (m³)'] for idx in emprestimos_concentrados_idx)
        }
    }
    
    return {
        'cft': cft_distribution,
        'ca': ca_distribution,
        'bota_fora': bota_fora,
        'distances': distances_df,
        'remaining_cft': remaining_cft,
        'remaining_ca': remaining_ca,
        'momento_total': momento_total,
        'dmt': dmt,
        'status': status_text,
        'emprestimos_laterais_nao_utilizados': emprestimos_laterais_nao_utilizados,
        'emprestimos_concentrados_nao_utilizados': emprestimos_concentrados_nao_utilizados,
        'total_emprestimo_lateral_nao_utilizado': total_emprestimo_lateral_nao_utilizado,
        'total_emprestimo_concentrado_nao_utilizado': total_emprestimo_concentrado_nao_utilizado,
        'total_emprestimo_nao_utilizado': total_emprestimo_nao_utilizado,
        'volume_por_tipo': volume_por_tipo,
        'favor_cortes': favor_cortes,
        'max_dist_cortes': max_dist_cortes,
        'max_dist_emprestimos_laterais': max_dist_emprestimos_laterais,
        'max_dist_emprestimos_concentrados': max_dist_emprestimos_concentrados,
        'fixed_allocations': fixed_allocations if fixed_allocations else []
    }

def generate_distribution_summary(result, origins_df, destinations_df):
    """
    Gera um resumo da distribuição otimizada
    """
    if result is None:
        return "Não foi possível encontrar uma solução factível."
    
    # Cria um resumo das distribuições
    cft_dist = result['cft']
    ca_dist = result['ca']
    bota_fora = result['bota_fora']
    dmt = result['dmt']
    
    # Total por tipo de material
    total_cft = cft_dist.sum().sum()
    total_ca = ca_dist.sum().sum()
    total_bota_fora = bota_fora.sum()
    
    summary = [
        f"Status da otimização: {result['status']}",
        f"Momento total de transporte: {result['momento_total']:.2f} m³·m",
        f"Distância média de transporte (DMT): {dmt:.2f} m",
        f"Volume CFT distribuído: {total_cft:.2f} m³",
        f"Volume CA distribuído: {total_ca:.2f} m³",
        f"Volume enviado para bota-fora: {total_bota_fora:.2f} m³"
    ]
    
    # Verifica se há volume não atendido
    remaining_cft = result['remaining_cft']
    remaining_ca = result['remaining_ca']
    
    total_remaining_cft = remaining_cft.sum()
    total_remaining_ca = remaining_ca.sum()
    
    if total_remaining_cft > 0:
        summary.append(f"Volume CFT não atendido: {total_remaining_cft:.2f} m³")
        # Detalha destinos não atendidos para CFT
        for d_idx in remaining_cft.index:
            if remaining_cft[d_idx] > 0:
                summary.append(f"  - Destino {d_idx}: {remaining_cft[d_idx]:.2f} m³")
    
    if total_remaining_ca > 0:
        summary.append(f"Volume CA não atendido: {total_remaining_ca:.2f} m³")
        # Detalha destinos não atendidos para CA
        for d_idx in remaining_ca.index:
            if remaining_ca[d_idx] > 0:
                summary.append(f"  - Destino {d_idx}: {remaining_ca[d_idx]:.2f} m³")
    
    # Informações sobre empréstimos não utilizados
    if 'volume_por_tipo' in result:
        volume_por_tipo = result['volume_por_tipo']
        
        # Cortes
        vpt_cortes = volume_por_tipo['cortes']
        perc_cortes_utilizado = (vpt_cortes['utilizado'] / vpt_cortes['total'] * 100) if vpt_cortes['total'] > 0 else 0
        summary.append(f"Cortes: {vpt_cortes['utilizado']:.2f} m³ utilizados de {vpt_cortes['total']:.2f} m³ ({perc_cortes_utilizado:.1f}%)")
        summary.append(f"Bota-fora de cortes: {vpt_cortes['bota_fora']:.2f} m³ ({vpt_cortes['bota_fora']/vpt_cortes['total']*100:.1f}% do volume de corte)")
        
        # Empréstimos laterais, se houver
        if 'emprestimos_laterais' in volume_por_tipo:
            vpt_emp_lat = volume_por_tipo['emprestimos_laterais']
            if vpt_emp_lat['total'] > 0:
                perc_lat_utilizado = (vpt_emp_lat['utilizado'] / vpt_emp_lat['total'] * 100) if vpt_emp_lat['total'] > 0 else 0
                summary.append(f"Empréstimos Laterais: {vpt_emp_lat['utilizado']:.2f} m³ utilizados de {vpt_emp_lat['total']:.2f} m³ ({perc_lat_utilizado:.1f}%)")
        
        # Empréstimos concentrados, se houver
        if 'emprestimos_concentrados' in volume_por_tipo:
            vpt_emp_conc = volume_por_tipo['emprestimos_concentrados']
            if vpt_emp_conc['total'] > 0:
                perc_conc_utilizado = (vpt_emp_conc['utilizado'] / vpt_emp_conc['total'] * 100) if vpt_emp_conc['total'] > 0 else 0
                summary.append(f"Empréstimos Concentrados: {vpt_emp_conc['utilizado']:.2f} m³ utilizados de {vpt_emp_conc['total']:.2f} m³ ({perc_conc_utilizado:.1f}%)")
    else:
        # Versão mais simples para resultados sem classificação detalhada
        total_emprestimo_nao_utilizado = result.get('total_emprestimo_nao_utilizado', 0)
        if total_emprestimo_nao_utilizado > 0:
            summary.append(f"Volume de empréstimo não utilizado: {total_emprestimo_nao_utilizado:.2f} m³")
    
    return "\n".join(summary)

def create_distribution_report(result, origins_df, destinations_df, filename=None):
    """
    Cria um relatório de distribuição em formato Excel
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
        filename: Nome do arquivo para salvar o relatório
    
    Returns:
        BytesIO object contendo o arquivo Excel
    """
    if result is None:
        return None
    
    # Cria um arquivo Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formato para células
        header_format = workbook.add_format({
            'bold': True, 
            'text_wrap': True, 
            'valign': 'top', 
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })
        
        # Status da otimização
        info_sheet = workbook.add_worksheet('Informações Gerais')
        
        # Título
        info_sheet.write(0, 0, 'Resumo da Distribuição de Terraplenagem', 
                        workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Data e hora
        now = datetime.datetime.now()
        info_sheet.write(1, 0, f'Relatório gerado em: {now.strftime("%d/%m/%Y %H:%M:%S")}')
        
        # Resumo geral
        row = 3
        info_sheet.write(row, 0, 'Status da Otimização', header_format)
        info_sheet.write(row, 1, result['status'], cell_format)
        row += 1
        
        info_sheet.write(row, 0, 'Momento Total de Transporte (m³·m)', header_format)
        info_sheet.write(row, 1, result['momento_total'], number_format)
        row += 1
        
        info_sheet.write(row, 0, 'Distância Média de Transporte (m)', header_format)
        info_sheet.write(row, 1, result['dmt'], number_format)
        row += 1
        
        # Resumo de volumes
        row += 1
        info_sheet.write(row, 0, 'Resumo de Volumes', 
                       workbook.add_format({'bold': True, 'font_size': 12}))
        row += 1
        
        # Cabeçalho de volumes
        info_sheet.write(row, 0, 'Tipo', header_format)
        info_sheet.write(row, 1, 'Volume Disponível (m³)', header_format)
        info_sheet.write(row, 2, 'Volume Distribuído (m³)', header_format)
        info_sheet.write(row, 3, 'Volume Não Utilizado (m³)', header_format)
        info_sheet.write(row, 4, 'Utilização (%)', header_format)
        row += 1
        
        # Volumes de origem
        total_origem = origins_df['Volume disponível (m³)'].sum()
        total_distribuido = result['cft'].sum().sum() + result['ca'].sum().sum()
        
        # Cortes x Empréstimos
        if 'volume_por_tipo' in result:
            vpt = result['volume_por_tipo']
            
            # Cortes
            corte_info = vpt['cortes']
            info_sheet.write(row, 0, 'Cortes', cell_format)
            info_sheet.write(row, 1, corte_info['total'], number_format)
            info_sheet.write(row, 2, corte_info['utilizado'], number_format)
            info_sheet.write(row, 3, corte_info['bota_fora'], number_format)
            if corte_info['total'] > 0:
                info_sheet.write(row, 4, corte_info['utilizado'] / corte_info['total'], 
                               workbook.add_format({'num_format': '0.0%', 'border': 1}))
            else:
                info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
            row += 1
            
            # Empréstimos Laterais, se houver
            if 'emprestimos_laterais' in vpt and vpt['emprestimos_laterais']['total'] > 0:
                emp_lat_info = vpt['emprestimos_laterais']
                info_sheet.write(row, 0, 'Empréstimos Laterais', cell_format)
                info_sheet.write(row, 1, emp_lat_info['total'], number_format)
                info_sheet.write(row, 2, emp_lat_info['utilizado'], number_format)
                info_sheet.write(row, 3, emp_lat_info['nao_utilizado'], number_format)
                if emp_lat_info['total'] > 0:
                    info_sheet.write(row, 4, emp_lat_info['utilizado'] / emp_lat_info['total'], 
                                   workbook.add_format({'num_format': '0.0%', 'border': 1}))
                else:
                    info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
                row += 1
                # Empréstimos Concentrados, se houver
            if 'emprestimos_concentrados' in vpt and vpt['emprestimos_concentrados']['total'] > 0:
                emp_conc_info = vpt['emprestimos_concentrados']
                info_sheet.write(row, 0, 'Empréstimos Concentrados', cell_format)
                info_sheet.write(row, 1, emp_conc_info['total'], number_format)
                info_sheet.write(row, 2, emp_conc_info['utilizado'], number_format)
                info_sheet.write(row, 3, emp_conc_info['nao_utilizado'], number_format)
                if emp_conc_info['total'] > 0:
                    info_sheet.write(row, 4, emp_conc_info['utilizado'] / emp_conc_info['total'], 
                                   workbook.add_format({'num_format': '0.0%', 'border': 1}))
                else:
                    info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
                row += 1
        else:
            # Versão simplificada se não houver classificação detalhada
            info_sheet.write(row, 0, 'Total Origens', cell_format)
            info_sheet.write(row, 1, total_origem, number_format)
            info_sheet.write(row, 2, total_distribuido, number_format)
            info_sheet.write(row, 3, total_origem - total_distribuido, number_format)
            if total_origem > 0:
                info_sheet.write(row, 4, total_distribuido / total_origem, 
                               workbook.add_format({'num_format': '0.0%', 'border': 1}))
            else:
                info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
            row += 1
        
        # Total geral
        info_sheet.write(row, 0, 'Total Geral', header_format)
        info_sheet.write(row, 1, total_origem, number_format)
        info_sheet.write(row, 2, total_distribuido, number_format)
        info_sheet.write(row, 3, total_origem - total_distribuido, number_format)
        if total_origem > 0:
            info_sheet.write(row, 4, total_distribuido / total_origem, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        row += 1
        
        # Volumes de destino
        row += 2
        info_sheet.write(row, 0, 'Resumo de Destinos', 
                       workbook.add_format({'bold': True, 'font_size': 12}))
        row += 1
        
        # Cabeçalho de destinos
        info_sheet.write(row, 0, 'Tipo', header_format)
        info_sheet.write(row, 1, 'Volume Necessário (m³)', header_format)
        info_sheet.write(row, 2, 'Volume Atendido (m³)', header_format)
        info_sheet.write(row, 3, 'Volume Faltante (m³)', header_format)
        info_sheet.write(row, 4, 'Atendimento (%)', header_format)
        row += 1
        
        # CFT
        total_cft_necessario = destinations_df['Volume CFT (m³)'].fillna(0).sum()
        total_cft_atendido = result['cft'].sum().sum()
        info_sheet.write(row, 0, 'CFT', cell_format)
        info_sheet.write(row, 1, total_cft_necessario, number_format)
        info_sheet.write(row, 2, total_cft_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_cft_necessario - total_cft_atendido), number_format)
        if total_cft_necessario > 0:
            info_sheet.write(row, 4, total_cft_atendido / total_cft_necessario, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        row += 1
        
        # CA
        total_ca_necessario = destinations_df['Volume CA (m³)'].fillna(0).sum()
        total_ca_atendido = result['ca'].sum().sum()
        info_sheet.write(row, 0, 'CA', cell_format)
        info_sheet.write(row, 1, total_ca_necessario, number_format)
        info_sheet.write(row, 2, total_ca_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_ca_necessario - total_ca_atendido), number_format)
        if total_ca_necessario > 0:
            info_sheet.write(row, 4, total_ca_atendido / total_ca_necessario, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        row += 1
        
        # Total geral
        total_destino = total_cft_necessario + total_ca_necessario
        total_atendido = total_cft_atendido + total_ca_atendido
        info_sheet.write(row, 0, 'Total Geral', header_format)
        info_sheet.write(row, 1, total_destino, number_format)
        info_sheet.write(row, 2, total_atendido, number_format)
        info_sheet.write(row, 3, max(0, total_destino - total_atendido), number_format)
        if total_destino > 0:
            info_sheet.write(row, 4, total_atendido / total_destino, 
                           workbook.add_format({'num_format': '0.0%', 'border': 1}))
        else:
            info_sheet.write(row, 4, 0, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Ajusta largura das colunas
        info_sheet.set_column('A:A', 22)
        info_sheet.set_column('B:E', 18)
        
        # Adiciona planilhas para distribuições detalhadas
        
        # Distribuição CFT
        cft_sheet = workbook.add_worksheet('Distribuição CFT')
        
        # Título
        cft_sheet.write(0, 0, 'Distribuição de Material CFT', 
                       workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distribuição
        row = 2
        
        # Cabeçalho com destinos
        cft_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            cft_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        cft_sheet.write(row, len(destinations_df.index)+1, "Total (m³)", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            cft_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            row_sum = 0
            for j, d_idx in enumerate(destinations_df.index):
                value = result['cft'].loc[o_idx, d_idx]
                cft_sheet.write(row+i, j+1, value if value > 0 else "", number_format)
                row_sum += value
            cft_sheet.write(row+i, len(destinations_df.index)+1, row_sum, number_format)
        
        # Totais por coluna
        total_row = row + len(origins_df.index)
        cft_sheet.write(total_row, 0, "Total (m³)", header_format)
        for j, d_idx in enumerate(destinations_df.index):
            col_sum = result['cft'][d_idx].sum()
            cft_sheet.write(total_row, j+1, col_sum, number_format)
        
        cft_sheet.write(total_row, len(destinations_df.index)+1, 
                      result['cft'].sum().sum(), number_format)
        
        # Ajusta larguras das colunas
        cft_sheet.set_column('A:A', 15)
        cft_sheet.set_column(1, len(destinations_df.index)+1, 12)
        
        # Distribuição CA
        ca_sheet = workbook.add_worksheet('Distribuição CA')
        
        # Título
        ca_sheet.write(0, 0, 'Distribuição de Material CA', 
                      workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distribuição
        row = 2
        
        # Cabeçalho com destinos
        ca_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            ca_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        ca_sheet.write(row, len(destinations_df.index)+1, "Total (m³)", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            ca_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            row_sum = 0
            for j, d_idx in enumerate(destinations_df.index):
                value = result['ca'].loc[o_idx, d_idx]
                ca_sheet.write(row+i, j+1, value if value > 0 else "", number_format)
                row_sum += value
            ca_sheet.write(row+i, len(destinations_df.index)+1, row_sum, number_format)
        
        # Totais por coluna
        total_row = row + len(origins_df.index)
        ca_sheet.write(total_row, 0, "Total (m³)", header_format)
        for j, d_idx in enumerate(destinations_df.index):
            col_sum = result['ca'][d_idx].sum()
            ca_sheet.write(total_row, j+1, col_sum, number_format)
        
        ca_sheet.write(total_row, len(destinations_df.index)+1, 
                     result['ca'].sum().sum(), number_format)
        
        # Ajusta larguras das colunas
        ca_sheet.set_column('A:A', 15)
        ca_sheet.set_column(1, len(destinations_df.index)+1, 12)
        
        # Distâncias
        dist_sheet = workbook.add_worksheet('Distâncias')
        
        # Título
        dist_sheet.write(0, 0, 'Matriz de Distâncias (m)', 
                        workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Escreve a matriz de distâncias
        row = 2
        
        # Cabeçalho com destinos
        dist_sheet.write(row, 0, 'Origem / Destino', header_format)
        for col, d_idx in enumerate(destinations_df.index):
            dist_sheet.write(row, col+1, f"Destino {d_idx}", header_format)
        row += 1
        
        # Linhas com origens
        for i, o_idx in enumerate(origins_df.index):
            dist_sheet.write(row+i, 0, f"Origem {o_idx}", cell_format)
            for j, d_idx in enumerate(destinations_df.index):
                dist_sheet.write(row+i, j+1, result['distances'].loc[o_idx, d_idx], number_format)
        
        # Ajusta larguras das colunas
        dist_sheet.set_column('A:A', 15)
        dist_sheet.set_column(1, len(destinations_df.index), 12)
        
        # Dados de entrada (origens)
        origem_sheet = workbook.add_worksheet('Origens')
        
        # Título
        origem_sheet.write(0, 0, 'Dados das Origens', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalhos das colunas
        for col, colname in enumerate(origins_df.columns):
            origem_sheet.write(row, col+1, colname, header_format)
        origem_sheet.write(row, 0, 'ID', header_format)
        row += 1
        
        # Dados das origens
        for i, (idx, o_row) in enumerate(origins_df.iterrows()):
            origem_sheet.write(row+i, 0, idx, cell_format)
            for j, col in enumerate(origins_df.columns):
                if pd.api.types.is_numeric_dtype(origins_df[col]):
                    origem_sheet.write(row+i, j+1, o_row[col] if pd.notna(o_row[col]) else "", number_format)
                else:
                    origem_sheet.write(row+i, j+1, o_row[col] if pd.notna(o_row[col]) else "", cell_format)
        
        # Adiciona bota-fora
        bota_fora_col = len(origins_df.columns) + 1
        origem_sheet.write(row-1, bota_fora_col, "Bota-Fora (m³)", header_format)
        
        for i, (idx, bf) in enumerate(result['bota_fora'].items()):
            if bf > 0:
                origem_sheet.write(row+i, bota_fora_col, bf, number_format)
            else:
                origem_sheet.write(row+i, bota_fora_col, "", number_format)
        
        # Ajusta larguras das colunas
        origem_sheet.set_column('A:A', 8)
        origem_sheet.set_column(1, len(origins_df.columns)+1, 15)
        
        # Dados de entrada (destinos)
        destino_sheet = workbook.add_worksheet('Destinos')
        
        # Título
        destino_sheet.write(0, 0, 'Dados dos Destinos', 
                           workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalhos das colunas
        for col, colname in enumerate(destinations_df.columns):
            destino_sheet.write(row, col+1, colname, header_format)
        destino_sheet.write(row, 0, 'ID', header_format)
        row += 1
        
        # Dados dos destinos
        for i, (idx, d_row) in enumerate(destinations_df.iterrows()):
            destino_sheet.write(row+i, 0, idx, cell_format)
            for j, col in enumerate(destinations_df.columns):
                if pd.api.types.is_numeric_dtype(destinations_df[col]):
                    destino_sheet.write(row+i, j+1, d_row[col] if pd.notna(d_row[col]) else "", number_format)
                else:
                    destino_sheet.write(row+i, j+1, d_row[col] if pd.notna(d_row[col]) else "", cell_format)
        
        # Adiciona colunas para verificação de atendimento
        col_offset = len(destinations_df.columns) + 1
        
        # CFT Atendido
        destino_sheet.write(row-1, col_offset, "CFT Atendido (m³)", header_format)
        # CFT Restante
        destino_sheet.write(row-1, col_offset+1, "CFT Faltante (m³)", header_format)
        # CA Atendido
        destino_sheet.write(row-1, col_offset+2, "CA Atendido (m³)", header_format)
        # CA Restante
        destino_sheet.write(row-1, col_offset+3, "CA Faltante (m³)", header_format)
        
        # Preenche valores
        for i, d_idx in enumerate(destinations_df.index):
            # CFT Atendido
            cft_atendido = result['cft'][d_idx].sum()
            destino_sheet.write(row+i, col_offset, cft_atendido, number_format)
            
            # CFT Restante
            cft_faltante = result['remaining_cft'][d_idx]
            destino_sheet.write(row+i, col_offset+1, cft_faltante if cft_faltante > 0 else "", number_format)
            
            # CA Atendido
            ca_atendido = result['ca'][d_idx].sum()
            destino_sheet.write(row+i, col_offset+2, ca_atendido, number_format)
            
            # CA Restante
            ca_faltante = result['remaining_ca'][d_idx]
            destino_sheet.write(row+i, col_offset+3, ca_faltante if ca_faltante > 0 else "", number_format)
        
        # Ajusta larguras das colunas
        destino_sheet.set_column('A:A', 8)
        destino_sheet.set_column(1, len(destinations_df.columns)+4, 15)
        
        # Resumo detalhado de origens
        detail_sheet = workbook.add_worksheet('Detalhamento por Origem')
        
        # Título
        detail_sheet.write(0, 0, 'Detalhamento de Uso por Origem', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalho
        detail_sheet.write(row, 0, 'ID', header_format)
        detail_sheet.write(row, 1, 'Tipo', header_format)
        detail_sheet.write(row, 2, 'Volume disponível (m³)', header_format)
        detail_sheet.write(row, 3, 'Volume para CFT (m³)', header_format)
        detail_sheet.write(row, 4, 'Volume para CA (m³)', header_format)
        detail_sheet.write(row, 5, 'Volume para Bota-fora (m³)', header_format)
        detail_sheet.write(row, 6, 'Volume não utilizado (m³)', header_format)
        detail_sheet.write(row, 7, 'Utilização (%)', header_format)
        row += 1
        
        # Dados
        for i, o_idx in enumerate(origins_df.index):
            vol_disp = origins_df.loc[o_idx, 'Volume disponível (m³)']
            vol_cft = result['cft'].loc[o_idx].sum()
            vol_ca = result['ca'].loc[o_idx].sum()
            vol_bf = result['bota_fora'][o_idx] if o_idx in result['bota_fora'] else 0
            
            # Calcula o volume não utilizado para empréstimos
            vol_nao_utilizado = 0
            if o_idx in result.get('emprestimos_laterais_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_laterais_nao_utilizados'][o_idx]
            elif o_idx in result.get('emprestimos_concentrados_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_concentrados_nao_utilizados'][o_idx]
            elif o_idx in result.get('emprestimos_nao_utilizados', {}):
                vol_nao_utilizado = result['emprestimos_nao_utilizados'][o_idx]
            
            # Calcula a utilização
            vol_utilizado = vol_cft + vol_ca
            utilizacao = vol_utilizado / vol_disp if vol_disp > 0 else 0
            
            # Escreve os dados
            detail_sheet.write(row+i, 0, o_idx, cell_format)
            detail_sheet.write(row+i, 1, origins_df.loc[o_idx, 'Tipo'], cell_format)
            detail_sheet.write(row+i, 2, vol_disp, number_format)
            detail_sheet.write(row+i, 3, vol_cft if vol_cft > 0 else "", number_format)
            detail_sheet.write(row+i, 4, vol_ca if vol_ca > 0 else "", number_format)
            detail_sheet.write(row+i, 5, vol_bf if vol_bf > 0 else "", number_format)
            detail_sheet.write(row+i, 6, vol_nao_utilizado if vol_nao_utilizado > 0 else "", number_format)
            detail_sheet.write(row+i, 7, utilizacao, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Ajusta larguras das colunas
        detail_sheet.set_column('A:A', 8)
        detail_sheet.set_column('B:B', 25)
        detail_sheet.set_column('C:G', 18)
        detail_sheet.set_column('H:H', 12)
        
        # Resumo detalhado de destinos
        dest_detail_sheet = workbook.add_worksheet('Detalhamento por Destino')
        
        # Título
        dest_detail_sheet.write(0, 0, 'Detalhamento de Atendimento por Destino', 
                                workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Cabeçalho
        dest_detail_sheet.write(row, 0, 'ID', header_format)
        dest_detail_sheet.write(row, 1, 'Centro de Massa (m)', header_format)
        dest_detail_sheet.write(row, 2, 'Volume CFT necessário (m³)', header_format)
        dest_detail_sheet.write(row, 3, 'Volume CFT atendido (m³)', header_format)
        dest_detail_sheet.write(row, 4, 'Volume CFT faltante (m³)', header_format)
        dest_detail_sheet.write(row, 5, 'Volume CA necessário (m³)', header_format)
        dest_detail_sheet.write(row, 6, 'Volume CA atendido (m³)', header_format)
        dest_detail_sheet.write(row, 7, 'Volume CA faltante (m³)', header_format)
        dest_detail_sheet.write(row, 8, 'Atendimento total (%)', header_format)
        row += 1
        
        # Dados
        for i, d_idx in enumerate(destinations_df.index):
            vol_cft_nec = destinations_df.loc[d_idx, 'Volume CFT (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CFT (m³)']) else 0
            vol_ca_nec = destinations_df.loc[d_idx, 'Volume CA (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CA (m³)']) else 0
            
            vol_cft_ate = result['cft'][d_idx].sum()
            vol_ca_ate = result['ca'][d_idx].sum()
            
            vol_cft_falt = max(0, vol_cft_nec - vol_cft_ate)
            vol_ca_falt = max(0, vol_ca_nec - vol_ca_ate)
            
            vol_total_nec = vol_cft_nec + vol_ca_nec
            vol_total_ate = vol_cft_ate + vol_ca_ate
            
            atendimento = vol_total_ate / vol_total_nec if vol_total_nec > 0 else 1
            
            # Escreve os dados
            dest_detail_sheet.write(row+i, 0, d_idx, cell_format)
            dest_detail_sheet.write(row+i, 1, destinations_df.loc[d_idx, 'Centro de Massa (m)'], number_format)
            dest_detail_sheet.write(row+i, 2, vol_cft_nec if vol_cft_nec > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 3, vol_cft_ate if vol_cft_ate > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 4, vol_cft_falt if vol_cft_falt > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 5, vol_ca_nec if vol_ca_nec > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 6, vol_ca_ate if vol_ca_ate > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 7, vol_ca_falt if vol_ca_falt > 0 else "", number_format)
            dest_detail_sheet.write(row+i, 8, atendimento, workbook.add_format({'num_format': '0.0%', 'border': 1}))
        
        # Ajusta larguras das colunas
        dest_detail_sheet.set_column('A:A', 8)
        dest_detail_sheet.set_column('B:H', 18)
        dest_detail_sheet.set_column('I:I', 15)
        
        # Dados da otimização
        params_sheet = workbook.add_worksheet('Parâmetros')
        
        # Título
        params_sheet.write(0, 0, 'Parâmetros da Otimização', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
        row = 2
        
        # Parâmetros utilizados
        params_sheet.write(row, 0, 'Parâmetro', header_format)
        params_sheet.write(row, 1, 'Valor', header_format)
        row += 1
        
        params_sheet.write(row, 0, 'Favorecimento de materiais de corte', cell_format)
        params_sheet.write(row, 1, 'Sim' if result.get('favor_cortes', False) else 'Não', cell_format)
        row += 1
        
        params_sheet.write(row, 0, 'Distância máxima para cortes (m)', cell_format)
        if result.get('max_dist_cortes') is not None:
            params_sheet.write(row, 1, result['max_dist_cortes'], number_format)
        else:
            params_sheet.write(row, 1, "Sem limite", cell_format)
        row += 1
        
        if 'max_dist_emprestimos_laterais' in result:
            params_sheet.write(row, 0, 'Distância máxima para empréstimos laterais (m)', cell_format)
            if result.get('max_dist_emprestimos_laterais') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos_laterais'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
            
            params_sheet.write(row, 0, 'Distância máxima para empréstimos concentrados (m)', cell_format)
            if result.get('max_dist_emprestimos_concentrados') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos_concentrados'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
        elif 'max_dist_emprestimos' in result:
            params_sheet.write(row, 0, 'Distância máxima para empréstimos (m)', cell_format)
            if result.get('max_dist_emprestimos') is not None:
                params_sheet.write(row, 1, result['max_dist_emprestimos'], number_format)
            else:
                params_sheet.write(row, 1, "Sem limite", cell_format)
            row += 1
        
        # Adiciona informações sobre alocações fixas, se houver
        fixed_allocs = result.get('fixed_allocations', [])
        if fixed_allocs:
            row += 2
            params_sheet.write(row, 0, 'Alocações Fixas Pré-definidas', 
                              workbook.add_format({'bold': True, 'font_size': 12}))
            row += 1
            
            params_sheet.write(row, 0, 'Origem', header_format)
            params_sheet.write(row, 1, 'Destino', header_format)
            params_sheet.write(row, 2, 'Volume (m³)', header_format)
            params_sheet.write(row, 3, 'Tipo', header_format)
            row += 1
            
            for i, alloc in enumerate(fixed_allocs):
                params_sheet.write(row+i, 0, alloc['origem'], cell_format)
                params_sheet.write(row+i, 1, alloc['destino'], cell_format)
                params_sheet.write(row+i, 2, float(alloc['volume']), number_format)
                params_sheet.write(row+i, 3, alloc['tipo'], cell_format)
        
        # Ajusta larguras das colunas
        params_sheet.set_column('A:A', 40)
        params_sheet.set_column('B:D', 15)
        
        # Quadro de Distribuição de Terraplenagem (consolidado)
        dist_quadro_sheet = workbook.add_worksheet('Quadro de Distribuição')

        # Título
        dist_quadro_sheet.write(0, 0, 'Quadro de Distribuição de Terraplenagem', 
                              workbook.add_format({'bold': True, 'font_size': 14}))

        # Cabeçalhos
        row = 2
        dist_quadro_sheet.write(row, 0, 'Origem', header_format)
        dist_quadro_sheet.write(row, 1, 'Destino', header_format)
        dist_quadro_sheet.write(row, 2, 'Utilização', header_format)
        dist_quadro_sheet.write(row, 3, 'Volume (m³)', header_format)
        dist_quadro_sheet.write(row, 4, 'DT (km)', header_format)
        dist_quadro_sheet.write(row, 5, 'Momento (m³.km)', header_format)
        row += 1

        # Dados de CFT
        all_movements = []
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                vol_cft = result['cft'].loc[o_idx, d_idx]
                if vol_cft > 0:
                    dt_metros = result['distances'].loc[o_idx, d_idx]
                    dt_km = dt_metros / 1000  # Converter para km
                    momento = vol_cft * dt_km  # Em m³.km
                    
                    all_movements.append({
                        'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                        'destino': d_idx,  # Já é o ID original pois é o índice do DataFrame
                        'utilizacao': 'CFT',
                        'volume': vol_cft,
                        'dt_km': dt_km,
                        'momento': momento,
                        'sort_key': 1  # Para ordenação: CFT primeiro
                    })

        # Dados de CA
        for o_idx in origins_df.index:
            for d_idx in destinations_df.index:
                vol_ca = result['ca'].loc[o_idx, d_idx]
                if vol_ca > 0:
                    dt_metros = result['distances'].loc[o_idx, d_idx]
                    dt_km = dt_metros / 1000  # Converter para km
                    momento = vol_ca * dt_km  # Em m³.km
                    
                    all_movements.append({
                        'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                        'destino': d_idx,  # Já é o ID original pois é o índice do DataFrame
                        'utilizacao': 'CA',
                        'volume': vol_ca,
                        'dt_km': dt_km,
                        'momento': momento,
                        'sort_key': 2  # Para ordenação: CA depois
                    })

        # Dados de Bota-fora
        for o_idx, vol_bf in result['bota_fora'].items():
            if vol_bf > 0:
                all_movements.append({
                    'origem': o_idx,  # Já é o ID original pois é o índice do DataFrame
                    'destino': "Bota-fora",
                    'utilizacao': 'Bota-fora',
                    'volume': vol_bf,
                    'dt_km': 0,  # Distância não aplicável para bota-fora
                    'momento': 0,  # Momento não aplicável para bota-fora
                    'sort_key': 3  # Para ordenação: Bota-fora por último
                })

        # Ordenar movimentos (primeiro por origem, depois por tipo de utilização)
        all_movements.sort(key=lambda x: (x['origem'], x['sort_key']))

        # Escrever todos os movimentos
        for i, mov in enumerate(all_movements):
            dist_quadro_sheet.write(row + i, 0, mov['origem'], cell_format)
            dist_quadro_sheet.write(row + i, 1, mov['destino'], cell_format)
            dist_quadro_sheet.write(row + i, 2, mov['utilizacao'], cell_format)
            dist_quadro_sheet.write(row + i, 3, mov['volume'], number_format)
            dist_quadro_sheet.write(row + i, 4, mov['dt_km'], workbook.add_format({'num_format': '0.000', 'border': 1}))
            dist_quadro_sheet.write(row + i, 5, mov['momento'], workbook.add_format({'num_format': '0.000', 'border': 1}))

        # Linha com totais
        total_row = row + len(all_movements)
        dist_quadro_sheet.write(total_row, 0, "TOTAL", header_format)
        dist_quadro_sheet.write(total_row, 1, "", header_format)
        dist_quadro_sheet.write(total_row, 2, "", header_format)
        dist_quadro_sheet.write(total_row, 3, sum(mov['volume'] for mov in all_movements), number_format)
        dist_quadro_sheet.write(total_row, 4, "", header_format)
        dist_quadro_sheet.write(total_row, 5, sum(mov['momento'] for mov in all_movements), 
                              workbook.add_format({'num_format': '0.000', 'border': 1, 'bold': True}))

        # Ajusta larguras das colunas
        dist_quadro_sheet.set_column('A:B', 15)
        dist_quadro_sheet.set_column('C:C', 12)
        dist_quadro_sheet.set_column('D:F', 16)
        
    # Se foi fornecido um nome de arquivo, salva nele
    if filename:
        with open(filename, 'wb') as f:
            f.write(output.getvalue())
    
    # Retorna o BytesIO para usar no Streamlit
    output.seek(0)
    return output

def export_optimization_results(result, origins_df, destinations_df):
    """
    Exporta os resultados da otimização para JSON
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        String JSON com os resultados
    """
    if result is None:
        return json.dumps({'error': 'Não foi possível encontrar uma solução factível.'})
    
    # Prepara os resultados para exportação
    export_data = {
        'status': result['status'],
        'momento_total': round(float(result['momento_total']), 2),
        'dmt': round(float(result['dmt']), 2),
        'favor_cortes': result['favor_cortes'],
        'max_dist_cortes': result['max_dist_cortes'],
        'parametros': {
            'favor_cortes': result['favor_cortes'],
            'max_dist_cortes': result['max_dist_cortes']
        }
    }
    
    # Adiciona parâmetros de empréstimos, se disponíveis
    if 'max_dist_emprestimos_laterais' in result:
        export_data['parametros']['max_dist_emprestimos_laterais'] = result['max_dist_emprestimos_laterais']
        export_data['parametros']['max_dist_emprestimos_concentrados'] = result['max_dist_emprestimos_concentrados']
    elif 'max_dist_emprestimos' in result:
        export_data['parametros']['max_dist_emprestimos'] = result['max_dist_emprestimos']
    
    # Distribuição CFT
    cft_dist = []
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            value = result['cft'].loc[o_idx, d_idx]
            if value > 0:
                cft_dist.append({
                    'origem': str(o_idx),
                    'destino': str(d_idx),
                    'volume': round(float(value), 2),
                    'distancia': round(float(result['distances'].loc[o_idx, d_idx]), 2),
                    'momento': round(float(value * result['distances'].loc[o_idx, d_idx]), 2)
                })
    
    # Distribuição CA
    ca_dist = []
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            value = result['ca'].loc[o_idx, d_idx]
            if value > 0:
                ca_dist.append({
                    'origem': str(o_idx),
                    'destino': str(d_idx),
                    'volume': round(float(value), 2),
                    'distancia': round(float(result['distances'].loc[o_idx, d_idx]), 2),
                    'momento': round(float(value * result['distances'].loc[o_idx, d_idx]), 2)
                })
    
    # Bota-fora
    bota_fora = []
    for o_idx, value in result['bota_fora'].items():
        if value > 0:
            bota_fora.append({
                'origem': str(o_idx),
                'volume': round(float(value), 2)
            })
    
    # Empréstimos não utilizados
    emprestimos_nao_utilizados = []
    
    # Verifica se temos a classificação avançada ou a simples
    if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
        for o_idx, value in result['emprestimos_laterais_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Lateral'
                })
        for o_idx, value in result['emprestimos_concentrados_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Concentrado'
                })
    elif 'emprestimos_nao_utilizados' in result:
        for o_idx, value in result['emprestimos_nao_utilizados'].items():
            if value > 0:
                emprestimos_nao_utilizados.append({
                    'origem': str(o_idx),
                    'volume': round(float(value), 2),
                    'tipo': 'Empréstimo'
                })
    
    # Volumes restantes
    volumes_restantes = []
    for d_idx in destinations_df.index:
        cft_rest = result['remaining_cft'].get(d_idx, 0)
        ca_rest = result['remaining_ca'].get(d_idx, 0)
        
        if cft_rest > 0 or ca_rest > 0:
            volumes_restantes.append({
                'destino': str(d_idx),
                'volume_cft': round(float(cft_rest), 2) if cft_rest > 0 else 0,
                'volume_ca': round(float(ca_rest), 2) if ca_rest > 0 else 0
            })
    
    # Resumo por origem
    resumo_origens = []
    for o_idx in origins_df.index:
        vol_disp = origins_df.loc[o_idx, 'Volume disponível (m³)']
        vol_cft = result['cft'].loc[o_idx].sum()
        vol_ca = result['ca'].loc[o_idx].sum()
        vol_bf = result['bota_fora'].get(o_idx, 0)
        
        # Volume não utilizado
        vol_nao_utilizado = 0
        if 'emprestimos_laterais_nao_utilizados' in result and o_idx in result['emprestimos_laterais_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_laterais_nao_utilizados'][o_idx]
        elif 'emprestimos_concentrados_nao_utilizados' in result and o_idx in result['emprestimos_concentrados_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_concentrados_nao_utilizados'][o_idx]
        elif 'emprestimos_nao_utilizados' in result and o_idx in result['emprestimos_nao_utilizados']:
            vol_nao_utilizado = result['emprestimos_nao_utilizados'][o_idx]
        
        # Tipo de origem
        tipo_origem = origins_df.loc[o_idx, 'Tipo']
        
        resumo_origens.append({
            'id': str(o_idx),
            'tipo': tipo_origem,
            'volume_disponivel': round(float(vol_disp), 2),
            'volume_cft': round(float(vol_cft), 2),
            'volume_ca': round(float(vol_ca), 2),
            'volume_bota_fora': round(float(vol_bf), 2),
            'volume_nao_utilizado': round(float(vol_nao_utilizado), 2),
            'utilizacao': round(float((vol_cft + vol_ca) / vol_disp), 4) if vol_disp > 0 else 0
        })
    
    # Resumo por destino
    resumo_destinos = []
    for d_idx in destinations_df.index:
        vol_cft_nec = destinations_df.loc[d_idx, 'Volume CFT (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CFT (m³)']) else 0
        vol_ca_nec = destinations_df.loc[d_idx, 'Volume CA (m³)'] if pd.notna(destinations_df.loc[d_idx, 'Volume CA (m³)']) else 0
        
        vol_cft_ate = result['cft'][d_idx].sum()
        vol_ca_ate = result['ca'][d_idx].sum()
        
        vol_total_nec = vol_cft_nec + vol_ca_nec
        vol_total_ate = vol_cft_ate + vol_ca_ate
        
        atendimento = vol_total_ate / vol_total_nec if vol_total_nec > 0 else 1
        
        resumo_destinos.append({
            'id': str(d_idx),
            'volume_cft_necessario': round(float(vol_cft_nec), 2),
            'volume_cft_atendido': round(float(vol_cft_ate), 2),
            'volume_cft_faltante': round(float(max(0, vol_cft_nec - vol_cft_ate)), 2),
            'volume_ca_necessario': round(float(vol_ca_nec), 2),
            'volume_ca_atendido': round(float(vol_ca_ate), 2),
            'volume_ca_faltante': round(float(max(0, vol_ca_nec - vol_ca_ate)), 2),
            'atendimento': round(float(atendimento), 4)
        })
    
    # Totais
    export_data['totais'] = {
        'volume_cft': round(float(result['cft'].sum().sum()), 2),
        'volume_ca': round(float(result['ca'].sum().sum()), 2),
        'volume_bota_fora': round(float(result['bota_fora'].sum()), 2),
        'volume_nao_utilizado': round(float(result.get('total_emprestimo_nao_utilizado', 0)), 2)
    }
    
    # Adicionar as alocações fixas
    if 'fixed_allocations' in result and result['fixed_allocations']:
        export_data['alocacoes_fixas'] = result['fixed_allocations']
    
    # Adiciona todos os dados ao objeto final
    export_data.update({
        'distribuicao_cft': cft_dist,
        'distribuicao_ca': ca_dist,
        'bota_fora': bota_fora,
        'emprestimos_nao_utilizados': emprestimos_nao_utilizados,
        'volumes_restantes': volumes_restantes,
        'resumo_origens': resumo_origens,
        'resumo_destinos': resumo_destinos
    })
    
    return json.dumps(export_data, indent=2)

def display_optimization_charts(result, origins_df, destinations_df):
    """
    Exibe gráficos sobre a distribuição otimizada usando Streamlit
    
    Args:
        result: Resultado da otimização
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    """
    if result is None:
        st.warning("Não é possível gerar gráficos para uma solução não factível.")
        return
    
    # Identifica os tipos de empréstimos a partir dos resultados
    if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
        emprestimos_laterais_idx = list(result['emprestimos_laterais_nao_utilizados'].keys())
        emprestimos_concentrados_idx = list(result['emprestimos_concentrados_nao_utilizados'].keys())
        emprestimos_idx = emprestimos_laterais_idx + emprestimos_concentrados_idx
    elif 'emprestimos_nao_utilizados' in result:
        emprestimos_idx = list(result['emprestimos_nao_utilizados'].keys())
        emprestimos_laterais_idx = []
        emprestimos_concentrados_idx = []
    else:
        # Tenta identificar a partir do tipo na origem
        emprestimos_idx = origins_df[
            origins_df['Tipo'].str.contains('Empr|empr|EMPR', regex=True)
        ].index.tolist()
        emprestimos_laterais_idx = origins_df[
            origins_df['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True)
        ].index.tolist()
        emprestimos_concentrados_idx = [idx for idx in emprestimos_idx 
                                      if idx not in emprestimos_laterais_idx]
    
    cortes_idx = [idx for idx in origins_df.index if idx not in emprestimos_idx]
    
    # Organiza dados para gráficos
    
    # 1. Gráfico de distribuição de volumes por tipo de origem
    st.subheader("Distribuição de Volumes por Tipo de Origem")
    
    # Prepara dados para o gráfico
    volume_cortes_cft = sum(result['cft'].loc[idx].sum() for idx in cortes_idx)
    volume_cortes_ca = sum(result['ca'].loc[idx].sum() for idx in cortes_idx)
    volume_cortes_bf = sum(result['bota_fora'].get(idx, 0) for idx in cortes_idx)
    
    if emprestimos_laterais_idx and emprestimos_concentrados_idx:
        # Versão com empréstimos separados
        volume_emp_lat_cft = sum(result['cft'].loc[idx].sum() for idx in emprestimos_laterais_idx)
        volume_emp_lat_ca = sum(result['ca'].loc[idx].sum() for idx in emprestimos_laterais_idx)
        volume_emp_lat_nu = result.get('total_emprestimo_lateral_nao_utilizado', 0)
        
        volume_emp_conc_cft = sum(result['cft'].loc[idx].sum() for idx in emprestimos_concentrados_idx)
        volume_emp_conc_ca = sum(result['ca'].loc[idx].sum() for idx in emprestimos_concentrados_idx)
        volume_emp_conc_nu = result.get('total_emprestimo_concentrado_nao_utilizado', 0)
        
        dados_grafico = {
            'Tipo': ['Cortes', 'Cortes', 'Cortes', 
                   'Emp. Laterais', 'Emp. Laterais', 'Emp. Laterais',
                   'Emp. Concentrados', 'Emp. Concentrados', 'Emp. Concentrados'],
            'Categoria': ['CFT', 'CA', 'Bota-fora',
                        'CFT', 'CA', 'Não Utilizado',
                        'CFT', 'CA', 'Não Utilizado'],
            'Volume (m³)': [volume_cortes_cft, volume_cortes_ca, volume_cortes_bf,
                          volume_emp_lat_cft, volume_emp_lat_ca, volume_emp_lat_nu,
                          volume_emp_conc_cft, volume_emp_conc_ca, volume_emp_conc_nu]
        }
    else:
        # Versão simplificada
        volume_emp_cft = sum(result['cft'].loc[idx].sum() for idx in emprestimos_idx)
        volume_emp_ca = sum(result['ca'].loc[idx].sum() for idx in emprestimos_idx)
        volume_emp_nu = result.get('total_emprestimo_nao_utilizado', 0)
        
        dados_grafico = {
            'Tipo': ['Cortes', 'Cortes', 'Cortes', 
                   'Empréstimos', 'Empréstimos', 'Empréstimos'],
            'Categoria': ['CFT', 'CA', 'Bota-fora',
                        'CFT', 'CA', 'Não Utilizado'],
            'Volume (m³)': [volume_cortes_cft, volume_cortes_ca, volume_cortes_bf,
                          volume_emp_cft, volume_emp_ca, volume_emp_nu]
        }
    
    chart_df = pd.DataFrame(dados_grafico)
    
    # Filtra categorias com volume zero para melhor visualização
    chart_df = chart_df[chart_df['Volume (m³)'] > 0]
    
    if not chart_df.empty:
        chart = st.bar_chart(
            chart_df,
            x="Tipo",
            y="Volume (m³)",
            color="Categoria"
        )
    
    # 2. Gráfico de distribuição de materiais por destino
    st.subheader("Distribuição de Materiais por Destino")
    
    dados_destinos = []
    for d_idx in destinations_df.index:
        vol_cft_atendido = result['cft'][d_idx].sum()
        vol_ca_atendido = result['ca'][d_idx].sum()
        vol_cft_faltante = max(0, result['remaining_cft'].get(d_idx, 0))
        vol_ca_faltante = max(0, result['remaining_ca'].get(d_idx, 0))
        
        if vol_cft_atendido > 0:
            dados_destinos.append({
                'Destino': f"Dest. {d_idx}",
                'Categoria': 'CFT Atendido',
                'Volume (m³)': vol_cft_atendido
            })
        if vol_ca_atendido > 0:
            dados_destinos.append({
                'Destino': f"Dest. {d_idx}",
                'Categoria': 'CA Atendido',
                'Volume (m³)': vol_ca_atendido
            })
        if vol_cft_faltante > 0:
            dados_destinos.append({
                'Destino': f"Dest. {d_idx}",
                'Categoria': 'CFT Faltante',
                'Volume (m³)': vol_cft_faltante
            })
        if vol_ca_faltante > 0:
            dados_destinos.append({
                'Destino': f"Dest. {d_idx}",
                'Categoria': 'CA Faltante',
                'Volume (m³)': vol_ca_faltante
            })
    
    dest_chart_df = pd.DataFrame(dados_destinos)
    
    if not dest_chart_df.empty:
        chart = st.bar_chart(
            dest_chart_df,
            x="Destino",
            y="Volume (m³)",
            color="Categoria"
        )
     # 3. Gráfico de utilização de origens
    st.subheader("Utilização das Origens")
    
    dados_utilizacao = []
    for o_idx in origins_df.index:
        vol_disp = origins_df.loc[o_idx, 'Volume disponível (m³)']
        vol_usado = result['cft'].loc[o_idx].sum() + result['ca'].loc[o_idx].sum()
        vol_bf = result['bota_fora'].get(o_idx, 0)
        vol_nu = 0
        
        if o_idx in emprestimos_idx:
            # Para empréstimos, o não utilizado é o volume que não foi distribuído
            if 'emprestimos_laterais_nao_utilizados' in result and o_idx in result['emprestimos_laterais_nao_utilizados']:
                vol_nu = result['emprestimos_laterais_nao_utilizados'][o_idx]
            elif 'emprestimos_concentrados_nao_utilizados' in result and o_idx in result['emprestimos_concentrados_nao_utilizados']:
                vol_nu = result['emprestimos_concentrados_nao_utilizados'][o_idx]
            elif 'emprestimos_nao_utilizados' in result and o_idx in result['emprestimos_nao_utilizados']:
                vol_nu = result['emprestimos_nao_utilizados'][o_idx]
        else:
            # Para cortes, o não utilizado vai para bota-fora
            vol_bf = result['bota_fora'].get(o_idx, 0)
        
        # Calcula a utilização percentual
        utilizacao = vol_usado / vol_disp if vol_disp > 0 else 0
        
        dados_utilizacao.append({
            'Origem': f"Origem {o_idx}",
            'Utilização (%)': utilizacao * 100,
            'Tipo': 'Corte' if o_idx in cortes_idx else 
                   'Emp. Lateral' if o_idx in emprestimos_laterais_idx else 
                   'Emp. Concentrado' if o_idx in emprestimos_concentrados_idx else
                   'Empréstimo',
            'Volume Disponível (m³)': vol_disp,
            'Volume Usado (m³)': vol_usado,
            'Volume Bota-fora/Não Util. (m³)': vol_bf if o_idx in cortes_idx else vol_nu
        })
    
    util_chart_df = pd.DataFrame(dados_utilizacao)
    
    # Ordena por utilização
    util_chart_df = util_chart_df.sort_values('Utilização (%)', ascending=False)
    
    if not util_chart_df.empty:
        chart = st.bar_chart(
            util_chart_df,
            x="Origem",
            y="Utilização (%)",
            color="Tipo"
        )
    
    # 4. Gráfico de histograma de distância média
    st.subheader("Histograma de Distâncias de Transporte")
    
    dados_distancia = []
    for o_idx in origins_df.index:
        for d_idx in destinations_df.index:
            vol_cft = result['cft'].loc[o_idx, d_idx]
            vol_ca = result['ca'].loc[o_idx, d_idx]
            
            if vol_cft > 0:
                dados_distancia.append({
                    'Distância (m)': result['distances'].loc[o_idx, d_idx],
                    'Volume (m³)': vol_cft,
                    'Material': 'CFT',
                    'Tipo Origem': 'Corte' if o_idx in cortes_idx else 
                               'Emp. Lateral' if o_idx in emprestimos_laterais_idx else 
                               'Emp. Concentrado' if o_idx in emprestimos_concentrados_idx else
                               'Empréstimo'
                })
            
            if vol_ca > 0:
                dados_distancia.append({
                    'Distância (m)': result['distances'].loc[o_idx, d_idx],
                    'Volume (m³)': vol_ca,
                    'Material': 'CA',
                    'Tipo Origem': 'Corte' if o_idx in cortes_idx else 
                               'Emp. Lateral' if o_idx in emprestimos_laterais_idx else 
                               'Emp. Concentrado' if o_idx in emprestimos_concentrados_idx else
                               'Empréstimo'
                })
    
    dist_chart_df = pd.DataFrame(dados_distancia)
    
    if not dist_chart_df.empty:
        # Histograma de distância
        hist_values = dist_chart_df['Distância (m)'].tolist()
        
        if hist_values:
            st.write(f"Distância Média de Transporte: {result['dmt']:.2f} m")
            st.write(f"Distância Mínima: {min(hist_values):.2f} m")
            st.write(f"Distância Máxima: {max(hist_values):.2f} m")
            
            # Criar histograma com base no volume
            # Agrupa por faixas de distância
            dist_chart_df['Faixa de Distância'] = pd.cut(
                dist_chart_df['Distância (m)'], 
                bins=10,
                include_lowest=True
            )
            
            # Calcula o volume por faixa
            volume_por_faixa = dist_chart_df.groupby(['Faixa de Distância', 'Tipo Origem'])['Volume (m³)'].sum().reset_index()
            
            # Converte a faixa para string para melhor visualização
            volume_por_faixa['Faixa de Distância'] = volume_por_faixa['Faixa de Distância'].astype(str)
            
            chart = st.bar_chart(
                volume_por_faixa,
                x="Faixa de Distância",
                y="Volume (m³)",
                color="Tipo Origem"
            )

def validate_input_data(origins_df, destinations_df):
    """
    Valida os dados de entrada e corrige problemas comuns
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        Tuple com (origins_df, destinations_df) validados e corrigidos,
        lista de mensagens de aviso
    """
    warnings = []
    
    # Cria cópias para não modificar os originais
    origins = origins_df.copy()
    destinations = destinations_df.copy()
    
    # Verifica e corrige valores nulos
    for col in ["Volume disponível (m³)", "ISC", "Centro de Massa (m)"]:
        if col in origins.columns and origins[col].isna().any():
            count_nulls = origins[col].isna().sum()
            warnings.append(f"Encontrados {count_nulls} valores nulos na coluna '{col}' das origens. Serão substituídos.")
            
            if col == "Volume disponível (m³)":
                origins[col] = origins[col].fillna(0)
            elif col == "ISC":
                origins[col] = origins[col].fillna(0)
            elif col == "Centro de Massa (m)":
                # Usa a média ou 0 se não houver valores
                fill_value = origins[col].mean() if not origins[col].isna().all() else 0
                origins[col] = origins[col].fillna(fill_value)
    
    # Verifica e corrige valores nulos nos destinos
    for col in ["Volume CFT (m³)", "Volume CA (m³)", "Centro de Massa (m)"]:
        if col in destinations.columns and destinations[col].isna().any():
            count_nulls = destinations[col].isna().sum()
            warnings.append(f"Encontrados {count_nulls} valores nulos na coluna '{col}' dos destinos. Serão substituídos.")
            
            if col in ["Volume CFT (m³)", "Volume CA (m³)"]:
                destinations[col] = destinations[col].fillna(0)
            elif col == "Centro de Massa (m)":
                # Usa a média ou 0 se não houver valores
                fill_value = destinations[col].mean() if not destinations[col].isna().all() else 0
                destinations[col] = destinations[col].fillna(fill_value)
    
    # Verifica se a coluna de ISC mínimo existe, se não, cria
    if "ISC mínimo exigido" not in destinations.columns:
        destinations["ISC mínimo exigido"] = 0
        warnings.append("Coluna 'ISC mínimo exigido' não encontrada nos destinos. Criada com valor 0.")
    
    # Verifica se a coluna de tipo existe nas origens, se não, tenta inferir
    if "Tipo" not in origins.columns:
        # Tenta inferir o tipo pela descrição ou outras colunas
        origins["Tipo"] = "Corte"  # Valor padrão
        warnings.append("Coluna 'Tipo' não encontrada nas origens. Todos os materiais serão considerados como 'Corte'.")
    
    # Verifica valores negativos em volumes
    if (origins["Volume disponível (m³)"] < 0).any():
        count_neg = (origins["Volume disponível (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume de origens. Serão substituídos por 0.")
        origins.loc[origins["Volume disponível (m³)"] < 0, "Volume disponível (m³)"] = 0
    
    if "Volume CFT (m³)" in destinations.columns and (destinations["Volume CFT (m³)"] < 0).any():
        count_neg = (destinations["Volume CFT (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume CFT de destinos. Serão substituídos por 0.")
        destinations.loc[destinations["Volume CFT (m³)"] < 0, "Volume CFT (m³)"] = 0
    
    if "Volume CA (m³)" in destinations.columns and (destinations["Volume CA (m³)"] < 0).any():
        count_neg = (destinations["Volume CA (m³)"] < 0).sum()
        warnings.append(f"Encontrados {count_neg} valores negativos no volume CA de destinos. Serão substituídos por 0.")
        destinations.loc[destinations["Volume CA (m³)"] < 0, "Volume CA (m³)"] = 0
    
    return origins, destinations, warnings

def prepare_data_for_optimization(origins_df, destinations_df):
    """
    Prepara os dados para otimização, garantindo que estejam no formato correto
    
    Args:
        origins_df: DataFrame com dados das origens
        destinations_df: DataFrame com dados dos destinos
    
    Returns:
        Tuple com (origins_df, destinations_df) prontos para otimização
    """
    # Valida e corrige os dados
    origins, destinations, warnings = validate_input_data(origins_df, destinations_df)
    
    # REMOVIDO: Não converter índices para string, pois já são os IDs corretos
    # origins.index = origins.index.astype(str)
    # destinations.index = destinations.index.astype(str)
    
    # Exibe avisos no Streamlit, se houver
    for warning in warnings:
        st.warning(warning)
    
    return origins, destinations

def create_interface():
    """
    Cria a interface do Streamlit para o sistema de otimização
    """
    st.set_page_config(page_title="Otimização de Distribuição de Terraplenagem", layout="wide")
    
    st.title("Sistema de Distribuição de Terraplenagem com Otimização Automática")
    
    with st.sidebar:
        st.header("Opções")
        
        tab = st.radio("Selecione a operação:", 
                     ["Carregar Dados", "Visualizar Dados", "Configurar Otimização", 
                      "Executar Otimização", "Visualizar Resultados", "Exportar"])
        # Inicializa estados da sessão, se não existirem
    if 'origins_df' not in st.session_state:
        st.session_state.origins_df = None
    if 'destinations_df' not in st.session_state:
        st.session_state.destinations_df = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'fixed_allocations' not in st.session_state:
        st.session_state.fixed_allocations = []
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = None
    
    if tab == "Carregar Dados":
        st.header("Carregar Dados de Origem e Destino")
        
        st.subheader("1. Dados das Origens")
        st.write("""
        Carregue um arquivo Excel ou CSV com os dados das origens (cortes e empréstimos).
        
        O arquivo deve ter as seguintes colunas:
        - Tipo: Descrição do tipo de origem (Corte, Empréstimo, Empréstimo Lateral, etc.)
        - Centro de Massa (m): Posição (em metros) do centro de massa
        - Volume disponível (m³): Volume total disponível
        - ISC: Índice de Suporte Califórnia do material
        - DT Fixo (m) (opcional): Distância adicional fixa para empréstimos
        """)
        
        origins_file = st.file_uploader("Escolha o arquivo com os dados das origens", 
                                      type=["xlsx", "xls", "csv"], key="origins_uploader")
        
        if origins_file is not None:
            try:
                if origins_file.name.endswith('.csv'):
                    origins_df = pd.read_csv(origins_file, index_col="ID")  # Usar ID como índice
                else:
                    origins_df = pd.read_excel(origins_file, index_col="ID")  # Usar ID como índice
                
                # Verifica colunas mínimas necessárias
                required_cols = ["Tipo", "Centro de Massa (m)", "Volume disponível (m³)", "ISC"]
                missing_cols = [col for col in required_cols if col not in origins_df.columns]
                
                if missing_cols:
                    st.error(f"Colunas obrigatórias não encontradas no arquivo de origens: {', '.join(missing_cols)}")
                else:
                    st.session_state.origins_df = origins_df
                    st.success(f"Dados carregados com sucesso: {len(origins_df)} origens encontradas")
                    st.write(origins_df)
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {str(e)}")
        
        st.subheader("2. Dados dos Destinos")
        st.write("""
        Carregue um arquivo Excel ou CSV com os dados dos destinos.
        
        O arquivo deve ter as seguintes colunas:
        - Centro de Massa (m): Posição (em metros) do centro de massa
        - Volume CFT (m³): Volume necessário para corpo de fundação terroso
        - Volume CA (m³): Volume necessário para corpo do aterro
        - ISC mínimo exigido: ISC mínimo exigido para o material de CFT
        """)
        
        destinations_file = st.file_uploader("Escolha o arquivo com os dados dos destinos", 
                                          type=["xlsx", "xls", "csv"], key="destinations_uploader")
        
        if destinations_file is not None:
            try:
                if destinations_file.name.endswith('.csv'):
                    destinations_df = pd.read_csv(destinations_file, index_col="ID")  # Usar ID como índice
                else:
                    destinations_df = pd.read_excel(destinations_file, index_col="ID")  # Usar ID como índice
                
                # Verifica colunas mínimas necessárias
                required_cols = ["Centro de Massa (m)", "Volume CFT (m³)", "Volume CA (m³)"]
                missing_cols = [col for col in required_cols if col not in destinations_df.columns]
                
                if missing_cols:
                    st.error(f"Colunas obrigatórias não encontradas no arquivo de destinos: {', '.join(missing_cols)}")
                else:
                    st.session_state.destinations_df = destinations_df
                    st.success(f"Dados carregados com sucesso: {len(destinations_df)} destinos encontrados")
                    st.write(destinations_df)
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {str(e)}")
    
    elif tab == "Visualizar Dados":
        st.header("Visualizar Dados Carregados")
        
        if st.session_state.origins_df is not None and st.session_state.destinations_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dados das Origens")
                st.dataframe(st.session_state.origins_df)
                
                # Resumo das origens
                st.subheader("Resumo das Origens")
                origens_por_tipo = st.session_state.origins_df.groupby('Tipo').agg({
                    'Volume disponível (m³)': 'sum',
                    'ISC': 'mean'
                })
                origens_por_tipo = origens_por_tipo.reset_index()
                origens_por_tipo.columns = ['Tipo', 'Volume Total (m³)', 'ISC Médio']
                st.dataframe(origens_por_tipo)
                
                # Gráfico de volumes por tipo
                st.subheader("Volume por Tipo de Origem")
                chart_data = pd.DataFrame({
                    'Tipo': st.session_state.origins_df['Tipo'],
                    'Volume (m³)': st.session_state.origins_df['Volume disponível (m³)']
                })
                st.bar_chart(chart_data, x='Tipo', y='Volume (m³)')
            
            with col2:
                st.subheader("Dados dos Destinos")
                st.dataframe(st.session_state.destinations_df)
                
                # Resumo dos destinos
                st.subheader("Resumo dos Destinos")
                total_cft = st.session_state.destinations_df['Volume CFT (m³)'].fillna(0).sum()
                total_ca = st.session_state.destinations_df['Volume CA (m³)'].fillna(0).sum()
                
                resumo_destinos = pd.DataFrame({
                    'Tipo': ['CFT', 'CA', 'Total'],
                    'Volume Total (m³)': [total_cft, total_ca, total_cft + total_ca]
                })
                st.dataframe(resumo_destinos)
                
                # Gráfico de volumes por destino
                st.subheader("Volume por Destino")
                chart_data = pd.DataFrame()
                
                for i, row in st.session_state.destinations_df.iterrows():
                    if pd.notna(row['Volume CFT (m³)']) and row['Volume CFT (m³)'] > 0:
                        chart_data = pd.concat([chart_data, pd.DataFrame({
                            'Destino': [f"Destino {i}"],
                            'Tipo': ['CFT'],
                            'Volume (m³)': [row['Volume CFT (m³)']]
                        })])
                    
                    if pd.notna(row['Volume CA (m³)']) and row['Volume CA (m³)'] > 0:
                        chart_data = pd.concat([chart_data, pd.DataFrame({
                            'Destino': [f"Destino {i}"],
                            'Tipo': ['CA'],
                            'Volume (m³)': [row['Volume CA (m³)']]
                        })])
                
                st.bar_chart(chart_data, x='Destino', y='Volume (m³)', color='Tipo')
            
            # Verificação de factibilidade
            st.subheader("Verificação de Factibilidade")
            is_feasible, message = check_feasibility(st.session_state.origins_df, st.session_state.destinations_df)
            
            if is_feasible:
                st.success(message)
                
                total_origem = st.session_state.origins_df['Volume disponível (m³)'].sum()
                total_destino = (st.session_state.destinations_df['Volume CFT (m³)'].fillna(0) + 
                                st.session_state.destinations_df['Volume CA (m³)'].fillna(0)).sum()
                
                st.write(f"Volume total disponível nas origens: {total_origem:.2f} m³")
                st.write(f"Volume total necessário nos destinos: {total_destino:.2f} m³")
                st.write(f"Diferença (volume excedente): {total_origem - total_destino:.2f} m³")
            else:
                st.error(message)
                
                total_origem = st.session_state.origins_df['Volume disponível (m³)'].sum()
                total_destino = (st.session_state.destinations_df['Volume CFT (m³)'].fillna(0) + 
                                st.session_state.destinations_df['Volume CA (m³)'].fillna(0)).sum()
                
                st.write(f"Volume total disponível nas origens: {total_origem:.2f} m³")
                st.write(f"Volume total necessário nos destinos: {total_destino:.2f} m³")
                st.write(f"Déficit de volume: {total_destino - total_origem:.2f} m³")
        else:
            st.warning("Carregue os dados das origens e destinos antes de visualizar.")

    elif tab == "Configurar Otimização":
        st.header("Configurar Parâmetros da Otimização")
        
        if st.session_state.origins_df is not None and st.session_state.destinations_df is not None:
            # Parametrização básica
            st.subheader("Parâmetros Básicos")
            
            favor_cortes = st.checkbox("Favorecer materiais de corte sobre empréstimos", value=True)
            st.write("Quando ativado, o sistema priorizará o uso de materiais de corte antes de utilizar empréstimos.")
            
            time_limit = st.slider("Tempo limite para otimização (segundos)", min_value=60, max_value=3600, value=1800, step=60)
            st.write(f"O otimizador tentará encontrar a melhor solução dentro de {time_limit} segundos.")
            
            # Configuração de distâncias máximas
            st.subheader("Distâncias Máximas de Transporte")
            
            use_max_dist = st.checkbox("Limitar distâncias de transporte", value=False)
            max_dist_cortes = None
            max_dist_emprestimos_laterais = None
            max_dist_emprestimos_concentrados = None
            
            # Verifica se existem tipos específicos de empréstimos
            origens = st.session_state.origins_df
            has_laterais = any(origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            has_concentrados = any(origens['Tipo'].str.contains('Empr|empr|EMPR', regex=True) & 
                                 ~origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            
            if use_max_dist:
                max_dist_cortes = st.number_input("Distância máxima para materiais de corte (m)", min_value=0.0, value=5000.0, step=100.0)
                
                if has_laterais and has_concentrados:
                    # Configuração separada para cada tipo
                    max_dist_emprestimos_laterais = st.number_input("Distância máxima para empréstimos laterais (m)", 
                                                                  min_value=0.0, value=2000.0, step=100.0)
                    max_dist_emprestimos_concentrados = st.number_input("Distância máxima para empréstimos concentrados (m)", 
                                                                     min_value=0.0, value=10000.0, step=100.0)
                else:
                    # Configuração única para todos os empréstimos
                    max_dist_emprestimos = st.number_input("Distância máxima para empréstimos (m)", 
                                                        min_value=0.0, value=5000.0, step=100.0)
                    max_dist_emprestimos_laterais = max_dist_emprestimos
                    max_dist_emprestimos_concentrados = max_dist_emprestimos
            
            # Alocações fixas
            st.subheader("Alocações Fixas")
            
            st.write("""
            Defina alocações fixas que o otimizador deve respeitar. 
            Essas alocações serão mantidas independentemente da otimização.
            """)
            
            # Botão para adicionar alocação fixa
            if st.button("Adicionar Alocação Fixa"):
                # Interface para adicionar alocação
                with st.form("add_fixed_allocation"):
                    st.write("Nova Alocação Fixa")
                    
                    # Lista de origens e destinos
                    origens_ids = st.session_state.origins_df.index.tolist()
                    destinos_ids = st.session_state.destinations_df.index.tolist()
                    
                    origem_id = st.selectbox("Origem", options=origens_ids)
                    destino_id = st.selectbox("Destino", options=destinos_ids)
                    tipo_material = st.selectbox("Tipo de Material", options=["CFT", "CA"])
                    volume = st.number_input("Volume (m³)", min_value=0.0, step=10.0)
                    
                    # Salvar alocação
                    if st.form_submit_button("Salvar Alocação"):
                        nova_alocacao = {
                            'origem': origem_id,  # Usar os IDs originais
                            'destino': destino_id,  # Usar os IDs originais
                            'tipo': tipo_material,
                            'volume': float(volume)
                        }
                        
                        if 'fixed_allocations' not in st.session_state:
                            st.session_state.fixed_allocations = []
                        
                        st.session_state.fixed_allocations.append(nova_alocacao)
                        st.success("Alocação fixa adicionada com sucesso!")
            
            # Exibir alocações fixas atuais
            if 'fixed_allocations' in st.session_state and st.session_state.fixed_allocations:
                st.write("Alocações Fixas Atuais:")
                
                # Converter para DataFrame para melhor visualização
                fixed_alloc_df = pd.DataFrame(st.session_state.fixed_allocations)
                st.dataframe(fixed_alloc_df)
                
                if st.button("Limpar Todas as Alocações Fixas"):
                    st.session_state.fixed_allocations = []
                    st.success("Todas as alocações fixas foram removidas.")
            else:
                st.info("Nenhuma alocação fixa definida.")
            
            # Salvar configuração
            st.subheader("Salvar Configuração")
            
            config_name = st.text_input("Nome da configuração", value="config_otimizacao")
            
            if st.button("Salvar Configuração"):
                try:
                    config = {
                        'favor_cortes': favor_cortes,
                        'time_limit': time_limit,
                        'use_max_dist': use_max_dist,
                        'max_dist_cortes': max_dist_cortes if use_max_dist else None,
                        'max_dist_emprestimos_laterais': max_dist_emprestimos_laterais if use_max_dist else None,
                        'max_dist_emprestimos_concentrados': max_dist_emprestimos_concentrados if use_max_dist else None,
                        'fixed_allocations': st.session_state.fixed_allocations if 'fixed_allocations' in st.session_state else []
                    }
                    
                    # Salvar como JSON
                    with open(f"{config_name}.json", "w") as f:
                        json.dump(config, f, indent=4)
                    
                    st.success(f"Configuração salva com sucesso em '{config_name}.json'")
                except Exception as e:
                    st.error(f"Erro ao salvar configuração: {str(e)}")
            
            # Carregar configuração existente
            st.subheader("Carregar Configuração")
            
            config_file = st.file_uploader("Escolha um arquivo de configuração", type=["json"], key="config_uploader")
            
            if config_file is not None:
                try:
                    config = json.load(config_file)
                    
                    # Atualizar estado da sessão com os valores do arquivo
                    if 'favor_cortes' in config:
                        st.session_state.favor_cortes = config['favor_cortes']
                    if 'time_limit' in config:
                        st.session_state.time_limit = config['time_limit']
                    if 'use_max_dist' in config:
                        st.session_state.use_max_dist = config['use_max_dist']
                    if 'max_dist_cortes' in config:
                        st.session_state.max_dist_cortes = config['max_dist_cortes']
                    if 'max_dist_emprestimos_laterais' in config:
                        st.session_state.max_dist_emprestimos_laterais = config['max_dist_emprestimos_laterais']
                    if 'max_dist_emprestimos_concentrados' in config:
                        st.session_state.max_dist_emprestimos_concentrados = config['max_dist_emprestimos_concentrados']
                    if 'fixed_allocations' in config:
                        st.session_state.fixed_allocations = config['fixed_allocations']
                    
                    st.success("Configuração carregada com sucesso!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erro ao carregar configuração: {str(e)}")
        else:
            st.warning("Carregue os dados das origens e destinos antes de configurar a otimização.")
    
    elif tab == "Executar Otimização":
        st.header("Executar Otimização de Distribuição")
        
        if st.session_state.origins_df is not None and st.session_state.destinations_df is not None:
            # Parâmetros da execução
            st.subheader("Parâmetros da Execução")
            
            favor_cortes = st.checkbox("Favorecer materiais de corte sobre empréstimos", value=True)
            
            time_limit = st.slider("Tempo limite para otimização (segundos)", 
                                 min_value=60, max_value=3600, value=1800, step=60)
            
            use_max_dist = st.checkbox("Limitar distâncias de transporte", value=False)
            max_dist_cortes = None
            max_dist_emprestimos_laterais = None
            max_dist_emprestimos_concentrados = None
            
            # Verifica se existem tipos específicos de empréstimos
            origens = st.session_state.origins_df
            has_laterais = any(origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            has_concentrados = any(origens['Tipo'].str.contains('Empr|empr|EMPR', regex=True) & 
                                 ~origens['Tipo'].str.contains('Lateral|lateral|LATERAL', regex=True))
            
            if use_max_dist:
                max_dist_cortes = st.number_input("Distância máxima para materiais de corte (m)", 
                                               min_value=0.0, value=5000.0, step=100.0)
                
                if has_laterais and has_concentrados:
                    # Configuração separada para cada tipo
                    max_dist_emprestimos_laterais = st.number_input(
                        "Distância máxima para empréstimos laterais (m)", 
                        min_value=0.0, value=2000.0, step=100.0
                    )
                    max_dist_emprestimos_concentrados = st.number_input(
                        "Distância máxima para empréstimos concentrados (m)", 
                        min_value=0.0, value=10000.0, step=100.0
                    )
                else:
                    # Configuração única para todos os empréstimos
                    max_dist_emprestimos = st.number_input(
                        "Distância máxima para empréstimos (m)", 
                        min_value=0.0, value=5000.0, step=100.0
                    )
                    max_dist_emprestimos_laterais = max_dist_emprestimos
                    max_dist_emprestimos_concentrados = max_dist_emprestimos
            
            # Mostrar alocações fixas, se houverem
            if 'fixed_allocations' in st.session_state and st.session_state.fixed_allocations:
                st.subheader("Alocações Fixas Configuradas")
                
                # Converter para DataFrame para melhor visualização
                fixed_alloc_df = pd.DataFrame(st.session_state.fixed_allocations)
                st.dataframe(fixed_alloc_df)
                
                # Opção para desabilitar temporariamente
                use_fixed_allocations = st.checkbox("Utilizar alocações fixas na otimização", value=True)
                if not use_fixed_allocations:
                    fixed_allocations = None
                else:
                    fixed_allocations = st.session_state.fixed_allocations
            else:
                fixed_allocations = None
            
            # Verificação de factibilidade
            is_feasible, message = check_feasibility(st.session_state.origins_df, st.session_state.destinations_df)
            
            if not is_feasible:
                st.error(f"O problema parece ser infactível: {message}")
                st.warning("Você ainda pode tentar executar a otimização, mas pode não encontrar uma solução válida.")
            
            # Botão para executar a otimização
            if st.button("Executar Otimização"):
                cortes_idx, emprestimos_laterais_idx, emprestimos_concentrados_idx = identify_emprestimo_types(
                    st.session_state.origins_df
                )
                
                st.write("Iniciando otimização... Isso pode levar alguns minutos.")
                
                # Inicializa barra de progresso
                progress_bar = st.progress(0)
                st.session_state.progress_bar = progress_bar
                
                try:
                    # Determinar qual função de otimização usar
                    if has_laterais and has_concentrados:
                        # Versão avançada com tipos específicos de empréstimos
                        result = optimize_distribution_advanced(
                            st.session_state.origins_df,
                            st.session_state.destinations_df,
                            time_limit=time_limit,
                            favor_cortes=favor_cortes,
                            max_dist_cortes=max_dist_cortes if use_max_dist else None,
                            max_dist_emprestimos_laterais=max_dist_emprestimos_laterais if use_max_dist else None,
                            max_dist_emprestimos_concentrados=max_dist_emprestimos_concentrados if use_max_dist else None,
                            fixed_allocations=fixed_allocations,
                            cortes_idx=cortes_idx,
                            emprestimos_laterais_idx=emprestimos_laterais_idx,
                            emprestimos_concentrados_idx=emprestimos_concentrados_idx
                        )
                    else:
                        # Versão padrão
                        result = optimize_distribution(
                            st.session_state.origins_df,
                            st.session_state.destinations_df,
                            time_limit=time_limit,
                            favor_cortes=favor_cortes,
                            max_dist_cortes=max_dist_cortes if use_max_dist else None,
                            max_dist_emprestimos=max_dist_emprestimos_laterais if use_max_dist else None,
                            fixed_allocations=fixed_allocations
                        )
                    
                    # Atualiza a barra de progresso para 100%
                    progress_bar.progress(100)
                    
                    # Armazena o resultado
                    st.session_state.optimization_result = result
                    
                    if result:
                        st.success("Otimização concluída com sucesso!")
                        
                        # Exibir resumo dos resultados
                        st.subheader("Resumo da Otimização")
                        
                        summary = generate_distribution_summary(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        
                        st.text(summary)
                        
                        # Botão para ir para visualização detalhada
                        if st.button("Ir para Visualização Detalhada"):
                            st.session_state.tab = "Visualizar Resultados"
                            st.experimental_rerun()
                    else:
                        st.error("A otimização não conseguiu encontrar uma solução factível!")
                        st.write("Tente ajustar os parâmetros ou verificar os dados de entrada.")
                
                except Exception as e:
                    st.error(f"Erro durante a otimização: {str(e)}")
                    st.write("Detalhes do erro:")
                    st.write(str(e))
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Carregue os dados das origens e destinos antes de executar a otimização.")
    
    elif tab == "Visualizar Resultados":
        st.header("Visualização dos Resultados da Otimização")
        
        if 'optimization_result' in st.session_state and st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            
            # Exibir status e métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", result['status'])
            with col2:
                st.metric("DMT (m)", f"{result['dmt']:.2f}")
            with col3:
                momento_total_km = result['momento_total'] / 1000  # Converter para km·m³
                st.metric("Momento Total (km·m³)", f"{momento_total_km:.2f}")
            
            # Tabs para diferentes visualizações
            result_tabs = st.tabs(["Resumo", "Gráficos", "Detalhes CFT", "Detalhes CA", "Bota-fora"])
            
            # Tab de Resumo
            with result_tabs[0]:
                st.subheader("Resumo da Distribuição")
                
                summary = generate_distribution_summary(
                    result, 
                    st.session_state.origins_df, 
                    st.session_state.destinations_df
                )
                
                st.text(summary)
                
                # Informações sobre os parâmetros utilizados
                st.subheader("Parâmetros Utilizados")
                
                params_text = [
                    f"Favorecimento de cortes: {'Sim' if result.get('favor_cortes') else 'Não'}"
                ]
                
                if result.get('max_dist_cortes') is not None:
                    params_text.append(f"Distância máxima para cortes: {result['max_dist_cortes']:.0f} m")
                
                if 'max_dist_emprestimos_laterais' in result and result['max_dist_emprestimos_laterais'] is not None:
                    params_text.append(f"Distância máxima para empréstimos laterais: {result['max_dist_emprestimos_laterais']:.0f} m")
                
                if 'max_dist_emprestimos_concentrados' in result and result['max_dist_emprestimos_concentrados'] is not None:
                    params_text.append(f"Distância máxima para empréstimos concentrados: {result['max_dist_emprestimos_concentrados']:.0f} m")
                
                if 'max_dist_emprestimos' in result and result['max_dist_emprestimos'] is not None:
                    params_text.append(f"Distância máxima para empréstimos: {result['max_dist_emprestimos']:.0f} m")
                
                if 'fixed_allocations' in result and result['fixed_allocations']:
                    params_text.append(f"Alocações fixas utilizadas: {len(result['fixed_allocations'])}")
                
                for param in params_text:
                    st.write(param)
            
            # Tab de Gráficos
            with result_tabs[1]:
                # Utilizar a função de gráficos criada anteriormente
                display_optimization_charts(
                    result,
                    st.session_state.origins_df,
                    st.session_state.destinations_df
                )
            
            # Tab de Detalhes CFT
            with result_tabs[2]:
                st.subheader("Distribuição Detalhada de CFT")
                
                # Mostrar a matriz de distribuição
                st.write("Matriz de distribuição CFT (m³):")
                st.dataframe(result['cft'])
                
                # Informação adicional sobre CFT
                total_cft_necessario = st.session_state.destinations_df['Volume CFT (m³)'].fillna(0).sum()
                total_cft_distribuido = result['cft'].sum().sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Volume CFT Necessário", f"{total_cft_necessario:.2f} m³")
                col2.metric("Volume CFT Distribuído", f"{total_cft_distribuido:.2f} m³")
                col3.metric("Atendimento", f"{(total_cft_distribuido/total_cft_necessario*100):.1f}%" if total_cft_necessario > 0 else "N/A")
                
                # Verificar se há destinos não atendidos
                remaining_cft = result['remaining_cft']
                if remaining_cft.sum() > 0:
                    st.warning("Alguns destinos não foram totalmente atendidos para CFT!")
                    # Listar destinos não atendidos
                    st.write("Destinos com déficit de CFT:")
                    
                    deficit_df = pd.DataFrame({
                        'Destino': remaining_cft.index[remaining_cft > 0],
                        'Volume Faltante (m³)': remaining_cft[remaining_cft > 0].values
                    })
                    
                    st.dataframe(deficit_df)
            
            # Tab de Detalhes CA
            with result_tabs[3]:
                st.subheader("Distribuição Detalhada de CA")
                
                # Mostrar a matriz de distribuição
                st.write("Matriz de distribuição CA (m³):")
                st.dataframe(result['ca'])
                
                # Informação adicional sobre CA
                total_ca_necessario = st.session_state.destinations_df['Volume CA (m³)'].fillna(0).sum()
                total_ca_distribuido = result['ca'].sum().sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Volume CA Necessário", f"{total_ca_necessario:.2f} m³")
                col2.metric("Volume CA Distribuído", f"{total_ca_distribuido:.2f} m³")
                col3.metric("Atendimento", f"{(total_ca_distribuido/total_ca_necessario*100):.1f}%" if total_ca_necessario > 0 else "N/A")
                
                # Verificar se há destinos não atendidos
                remaining_ca = result['remaining_ca']
                if remaining_ca.sum() > 0:
                    st.warning("Alguns destinos não foram totalmente atendidos para CA!")
                    
                    # Listar destinos não atendidos
                    st.write("Destinos com déficit de CA:")
                    
                    deficit_df = pd.DataFrame({
                        'Destino': remaining_ca.index[remaining_ca > 0],
                        'Volume Faltante (m³)': remaining_ca[remaining_ca > 0].values
                    })
                    
                    st.dataframe(deficit_df)
            
            # Tab de Bota-fora e material não utilizado
            with result_tabs[4]:
                st.subheader("Material para Bota-fora")
                
                # Material para bota-fora
                bota_fora = result['bota_fora']
                if bota_fora.sum() > 0:
                    st.write("Volumes direcionados para bota-fora:")
                    
                    bf_df = pd.DataFrame({
                        'Origem': bota_fora.index[bota_fora > 0],
                        'Volume (m³)': bota_fora[bota_fora > 0].values
                    })
                    
                    st.dataframe(bf_df)
                    
                    # Gráfico de bota-fora
                    if len(bf_df) > 0:
                        st.bar_chart(bf_df, x='Origem', y='Volume (m³)')
                else:
                    st.success("Não há material direcionado para bota-fora.")
                
                # Material de empréstimo não utilizado
                st.subheader("Material de Empréstimo Não Utilizado")
                
                # Verifica qual tipo de resultado temos (avançado ou simples)
                if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
                    emp_lat = result['emprestimos_laterais_nao_utilizados']
                    emp_conc = result['emprestimos_concentrados_nao_utilizados']
                    
                    # Criar DataFrames para cada tipo
                    emp_lat_df = pd.DataFrame({
                        'Origem': list(emp_lat.keys()),
                        'Volume (m³)': list(emp_lat.values()),
                        'Tipo': ['Lateral'] * len(emp_lat)
                    }) if emp_lat else pd.DataFrame()
                    
                    emp_conc_df = pd.DataFrame({
                        'Origem': list(emp_conc.keys()),
                        'Volume (m³)': list(emp_conc.values()),
                        'Tipo': ['Concentrado'] * len(emp_conc)
                    }) if emp_conc else pd.DataFrame()
                    
                    # Combinar os DataFrames
                    emp_df = pd.concat([emp_lat_df, emp_conc_df])
                    
                    if not emp_df.empty:
                        st.write("Volumes de empréstimo não utilizados:")
                        st.dataframe(emp_df)
                        
                        # Gráfico de empréstimos não utilizados
                        st.bar_chart(emp_df, x='Origem', y='Volume (m³)', color='Tipo')
                    else:
                        st.success("Todo o material de empréstimo foi utilizado.")
                    
                elif 'emprestimos_nao_utilizados' in result:
                    emp = result['emprestimos_nao_utilizados']
                    
                    if emp:
                        emp_df = pd.DataFrame({
                            'Origem': list(emp.keys()),
                            'Volume (m³)': list(emp.values())
                        })
                        
                        st.write("Volumes de empréstimo não utilizados:")
                        st.dataframe(emp_df)
                        
                        # Gráfico de empréstimos não utilizados
                        st.bar_chart(emp_df, x='Origem', y='Volume (m³)')
                    else:
                        st.success("Todo o material de empréstimo foi utilizado.")
                else:
                    st.info("Não há informações sobre material de empréstimo não utilizado.")
        else:
            st.warning("Execute a otimização antes de visualizar os resultados.")
    
    elif tab == "Exportar":
        st.header("Exportar Resultados da Otimização")
        
        if 'optimization_result' in st.session_state and st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            
            # Resumo da otimização
            st.subheader("Resumo da Otimização")
            
            summary = generate_distribution_summary(
                result, 
                st.session_state.origins_df, 
                st.session_state.destinations_df
            )
            
            st.text(summary)
            
            # Opções de exportação
            st.subheader("Exportar Relatório")
            
            # Escolha do formato
            export_format = st.radio("Escolha o formato de exportação:", 
                                   ["Excel (.xlsx)", "JSON (.json)"])
            
            # Nome do arquivo
            default_filename = f"distribuicao_terraplenagem_{uuid.uuid4().hex[:8]}"
            filename = st.text_input("Nome do arquivo (sem extensão):", value=default_filename)
            
            if st.button("Gerar e Baixar Relatório"):
                try:
                    if export_format == "Excel (.xlsx)":
                        # Gera relatório Excel
                        excel_file = create_distribution_report(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        
                        # Disponibiliza para download
                        st.download_button(
                            label="Baixar Relatório Excel",
                            data=excel_file,
                            file_name=f"{filename}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("Relatório Excel gerado com sucesso!")
                    
                    elif export_format == "JSON (.json)":
                        # Gera relatório JSON
                        json_data = export_optimization_results(
                            result, 
                            st.session_state.origins_df, 
                            st.session_state.destinations_df
                        )
                        
                        # Disponibiliza para download
                        st.download_button(
                            label="Baixar Relatório JSON",
                            data=json_data,
                            file_name=f"{filename}.json",
                            mime="application/json"
                        )
                        
                        st.success("Relatório JSON gerado com sucesso!")
                
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Exportação de dados para outros sistemas
            st.subheader("Dados para Outros Sistemas")
            
            st.write("""
            Os dados podem ser exportados para uso em outros sistemas ou para análise posterior.
            Selecione as opções abaixo para visualizar os dados em diferentes formatos.
            """)
            
            # Opções de visualização de dados
            data_options = st.multiselect(
                "Escolha os dados para visualizar:",
                options=[
                    "Matriz de distribuição CFT",
                    "Matriz de distribuição CA",
                    "Bota-fora",
                    "Material não utilizado",
                    "Distâncias"
                ],
                default=["Matriz de distribuição CFT", "Matriz de distribuição CA"]
            )
            
            if "Matriz de distribuição CFT" in data_options:
                st.write("### Matriz de distribuição CFT")
                st.dataframe(result['cft'])
                # Opção para baixar como CSV
                csv_cft = result['cft'].to_csv()
                st.download_button(
                    label="Baixar matriz CFT como CSV",
                    data=csv_cft,
                    file_name=f"{filename}_cft.csv",
                    mime="text/csv"
                )
            
            if "Matriz de distribuição CA" in data_options:
                st.write("### Matriz de distribuição CA")
                st.dataframe(result['ca'])
                
                # Opção para baixar como CSV
                csv_ca = result['ca'].to_csv()
                st.download_button(
                    label="Baixar matriz CA como CSV",
                    data=csv_ca,
                    file_name=f"{filename}_ca.csv",
                    mime="text/csv"
                )
            
            if "Bota-fora" in data_options:
                st.write("### Bota-fora")
                bf_df = pd.DataFrame({
                    'Origem': result['bota_fora'].index,
                    'Volume (m³)': result['bota_fora'].values
                })
                st.dataframe(bf_df)
                
                # Opção para baixar como CSV
                csv_bf = bf_df.to_csv(index=False)
                st.download_button(
                    label="Baixar dados de bota-fora como CSV",
                    data=csv_bf,
                    file_name=f"{filename}_bota_fora.csv",
                    mime="text/csv"
                )
            
            if "Material não utilizado" in data_options:
                st.write("### Material não utilizado")
                
                # Verifica qual tipo de resultado temos
                if 'emprestimos_laterais_nao_utilizados' in result and 'emprestimos_concentrados_nao_utilizados' in result:
                    # Versão avançada
                    emp_lat = result['emprestimos_laterais_nao_utilizados']
                    emp_conc = result['emprestimos_concentrados_nao_utilizados']
                    
                    # Criar DataFrames para cada tipo
                    emp_lat_df = pd.DataFrame({
                        'Origem': list(emp_lat.keys()),
                        'Volume (m³)': list(emp_lat.values()),
                        'Tipo': ['Lateral'] * len(emp_lat)
                    }) if emp_lat else pd.DataFrame()
                    
                    emp_conc_df = pd.DataFrame({
                        'Origem': list(emp_conc.keys()),
                        'Volume (m³)': list(emp_conc.values()),
                        'Tipo': ['Concentrado'] * len(emp_conc)
                    }) if emp_conc else pd.DataFrame()
                    
                    # Combinar os DataFrames
                    emp_df = pd.concat([emp_lat_df, emp_conc_df])
                elif 'emprestimos_nao_utilizados' in result:
                    # Versão simples
                    emp = result['emprestimos_nao_utilizados']
                    emp_df = pd.DataFrame({
                        'Origem': list(emp.keys()),
                        'Volume (m³)': list(emp.values()),
                        'Tipo': ['Empréstimo'] * len(emp)
                    }) if emp else pd.DataFrame()
                else:
                    emp_df = pd.DataFrame()
                
                if not emp_df.empty:
                    st.dataframe(emp_df)
                    # Opção para baixar como CSV
                    csv_emp = emp_df.to_csv(index=False)
                    st.download_button(
                        label="Baixar dados de material não utilizado como CSV",
                        data=csv_emp,
                        file_name=f"{filename}_nao_utilizado.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Não há material não utilizado.")
            
            if "Distâncias" in data_options:
                st.write("### Matriz de Distâncias")
                st.dataframe(result['distances'])
                
                # Opção para baixar como CSV
                csv_dist = result['distances'].to_csv()
                st.download_button(
                    label="Baixar matriz de distâncias como CSV",
                    data=csv_dist,
                    file_name=f"{filename}_distancias.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Execute a otimização antes de exportar os resultados.")

def main():
    """
    Função principal para executar o aplicativo Streamlit
    """
    create_interface()

if __name__ == "__main__":
    main()
       
         