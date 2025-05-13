# ─── (1) IMPORTS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
import pandas as pd
from gurobipy import Model, GRB, quicksum



# ─── (2) CARGA DE DATOS Y PARAMETROS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# 1. Cargar CSV de flota wide‐body
df_status_wb = pd.read_csv('Datos/Fleet_status_WB.csv')

# 2. Asignar IDs secuenciales según línea de archivo (1…n)
df_status_wb.insert(0, 'id', range(1, len(df_status_wb) + 1))

######################################################################################################################################################
# Motores para Comprar
######################################################################################################################################################

# Número de motores de repuesto que queremos comprar
n_extra = 3  

# ID de aviones (igual que antes)
n_aviones = len(df_status_wb)
P_WB     = list(range(1, n_aviones + 1))


######################################################################################################################################################

######################################################################################################################################################


# 3. Definir conjuntos de IDs

# IDs de motores extra: justo después de los aviones
I_extra = list(range(n_aviones + 1, n_aviones + n_extra + 1))
# IDs de motores totales: los propios (1…n_aviones) más los extra
I_WB = list(range(1, n_aviones + 1)) + I_extra
# Horizonte de Prueba
T    = list(range(1, 50))


# 4. Mapeos matricula ↔ id
mat2id = dict(zip(df_status_wb['matricula'], df_status_wb['id']))
id2mat = {i: m for m, i in mat2id.items()}
for i in I_extra:
    id2mat[i] = f"EXTRA_{i}"  

# 5. Leer ciclos diarios y convertir a semanales
df_cycles_wb = pd.read_csv('Datos/Operations_cycles_WB.csv')
df_cycles_wb['ac_norm'] = (
    df_cycles_wb['Aircraft']
    .str.replace(r'[-\s]', '', regex=True)
    .str.upper()
)
df_cycles_wb['cycles_per_week'] = df_cycles_wb['Value'] * 7
rate_map = dict(zip(df_cycles_wb['ac_norm'], df_cycles_wb['cycles_per_week']))

# 6. Normalizar operación en df_status
df_status_wb['op_norm'] = (
    df_status_wb['Operation']
    .str.replace(r'[-\s]', '', regex=True)
    .str.upper()
)

# 7. Construir c[id] = ciclos/semana para cada avión ID
c = {}
for _, row in df_status_wb.iterrows():
    op = row['op_norm']
    match = next((code for code in rate_map if op.startswith(code)), None)
    if match is None:
        raise KeyError(f"No encontré tasa para operación '{row['Operation']}'")
    c[row['id']] = rate_map[match]

# 8. Ciclos iniciales y umbrales
y0 = dict(zip(df_status_wb['id'], df_status_wb['cycles']))

df_max_wb = pd.read_csv('Datos/Max_cycles_WB.csv')
df_max_wb['code_norm'] = (
    df_max_wb['Aircraft_family']
    .str.replace(r'[-\s]', '', regex=True)
    .str.upper()
)
C_f = dict(zip(df_max_wb['code_norm'], df_max_wb['Max cycles']))

C = {}
for _, row in df_status_wb.iterrows():
    op = row['op_norm']
    fam = next((code for code in C_f if op.startswith(code)), None)
    if fam is None:
        raise KeyError(f"No umbral para operación '{row['Operation']}'")
    C[row['id']] = C_f[fam]


# 9. Parámetros económicos y constantes
df_motor_info = pd.read_csv('Datos/Motor_info.csv')
LeaseCost = int(df_motor_info.loc[df_motor_info['Action']=='Lease for week','Price'].iloc[0])
BuyCost   = int(df_motor_info.loc[df_motor_info['Action']=='Buy','Price'].iloc[0])
d         = 18
S0        = 0
M_max     = 5
M         = max(C.values())


# 10. Extender parametros para los motores extras
for i in I_extra:
    y0[i] = 0       # cero ciclos al inicio
    C[i] = M   # o algún valor suficientemente alto

print("✅ Datos cargados.")


# ─── (3) MODELO Y VARIABLES ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# 3.1. Instanciar el modelo
model = Model("WB_Maintenance")
model.Params.OutputFlag = 1      # 1 para ver salidas de Gurobi, 0 para silenciar

# 3.2. Variables de asignación y stock
# a[i,p,t] = 1 si el motor i está instalado en el avión p en la semana t
a = model.addVars(I_WB, P_WB, T, vtype=GRB.BINARY, name="a")
# s[i,t] = 1 si el motor i está en stock al inicio de la semana t
s = model.addVars(I_WB,   T, vtype=GRB.BINARY, name="s")

# 3.3. Variables de mantenimiento
# m[i,t] = 1 si el motor i inicia mantenimiento en la semana t
m = model.addVars(I_WB,   T, vtype=GRB.BINARY, name="m")
# r[i,t] = 1 si el motor i está en mantenimiento en la semana t
r = model.addVars(I_WB,   T, vtype=GRB.BINARY, name="r")

# 3.4. Variables de ciclos e inventario agregado
# y[i,t] = ciclos acumulados por el motor i al cierre de la semana t
y = model.addVars(I_WB,   T, vtype=GRB.CONTINUOUS, lb=0, name="y")
# S[t]   = inventario agregado de repuestos al inicio de la semana t
S = model.addVars(  T,    vtype=GRB.CONTINUOUS, lb=0, name="S")

# 3.5. Variables de cobertura con arrendo o compra
# ell[p,t] = 1 si el avión p opera con motor arrendado en la semana t
ell = model.addVars(P_WB, T, vtype=GRB.BINARY, name="ell")

# 3.6. Variable binaria: buy_extra[i,t] = 1 si compramos el motor extra i en la semana t
buy_extra = model.addVars(I_extra, T, vtype=GRB.BINARY, name="buy_extra")



# ─── (4) RESTRICCIONES ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────


# 4.0 (nuevo) Asignación inicial en t=1
for i in I_WB:
    if i <= n_aviones:
        # Motor “propio” i va al avión i
        model.addConstr(a[i, i, 1] == 1,    name=f"init_assign_{i}_{i}")
        model.addConstr(r[i, 1]   == 0,      name=f"init_no_maint_{i}")
        model.addConstr(s[i, 1]   == 0,      name=f"init_no_stock_{i}")
        # Asegurarse de que no esté en otro avión
        for p in P_WB:
            if p != i:
                model.addConstr(a[i, p, 1] == 0, name=f"init_noassign_{i}_{p}")

    else:
        # Motores extra: no asignados, no en mantención, no en stock al inicio
        for p in P_WB:
            model.addConstr(a[i, p, 1] == 0,  name=f"extra_noassign_{i}_{p}")
        model.addConstr(r[i, 1] == 0,          name=f"extra_no_maint_{i}")
        model.addConstr(s[i, 1] == 0,          name=f"extra_no_stock_{i}")

# 4.1 Estados de cada motor
#     Exclusividad de estado si es propio (a + r + s = 1)
#     Si no se han comprado no pueden pertenecer a ningún estado
for i in I_WB:
    for t in T:
        if i in I_extra:
            # hasta que lo compres suma=0; después suma=1
            model.addConstr(
                quicksum(a[i,p,t] for p in P_WB) + r[i,t] + s[i,t]
                == quicksum(buy_extra[i,τ] for τ in range(1, t+1)),
                name=f"exclusive_extra_{i}_{t}"
            )
        else:
            # motores propios siempre en un único estado
            model.addConstr(
                quicksum(a[i,p,t] for p in P_WB) + r[i,t] + s[i,t]
                == 1,
                name=f"exclusive_init_{i}_{t}"
            )



# 4.2 Cobertura revisada: cada avión p en t
#    o usa un motor (propio o extra comprado y asignado)
#    o lo arrenda.
for p in P_WB:
    for t in T:
        model.addConstr(
            quicksum(a[i, p, t] for i in I_WB)
            + ell[p, t]
            == 1,
            name=f"coverage_{p}_{t}"
        )


# 4.3 Duración del mantenimiento: r[i,t] = 1 exactamente d semanas tras m[i,t']
for i in I_WB:
    for t in T:
        model.addConstr(
            r[i, t] == quicksum(
                m[i, tau]
                for tau in range(max(1, t - d + 1), t + 1)
            ),
            name=f"maint_duration_{i}_{t}"
        )

# ─── (4.4) Acumulación con reset exacto tras mantención ─────────────────────
for i in I_WB:
    # Semana 1 ya la tenías bien:
    model.addConstr(
        y[i, 1] == y0[i] + quicksum(c[p] * a[i, p, 1] for p in P_WB),
        name=f"init_cycles_{i}"
    )
    
    # Semanas 2…T
    for t in T[1:]:
        # ciclos “esta semana”
        expr = quicksum(c[p] * a[i, p, t] for p in P_WB)
        
        if t - d >= 1:
            # Si la mantención que empezó en t−d acaba justo ahora,
            # m[i, t−d] = 1 identifica ese caso:
            model.addConstr(
                y[i, t]
                == (1 - m[i, t - d]) * (y[i, t-1] + expr)
                 + m[i, t - d] * expr,
                name=f"cycles_reset_exact_{i}_{t}"
            )
        else:
            # Aún no hay mantención previa que resetee:
            model.addConstr(
                y[i, t]
                == y[i, t-1] + expr,
                name=f"cycles_accum_{i}_{t}"
            )




# ─── (4.5) Límite estricto de ciclos tras mantenimiento ─────────────────
for i in I_WB:
    for t in T:
        model.addConstr(
            y[i, t]
            <= C[i]       # umbral del motor i
               + M * r[i, t]   # si está en mantención, permitimos hasta M ciclos
            ,
            name=f"cycle_limit_strict_{i}_{t}"
        )


# 4.6 Capacidad de inicios de mantenimiento: a lo sumo M_max motores por semana
for t in T:
    model.addConstr(
        quicksum(m[i, t] for i in I_WB) <= M_max,
        name=f"capacity_{t}"
    )

# 4.7 Flujo de inventario de repuestos
#     — En stock inicial (semana 1) usamos sólo compras de extra
model.addConstr(
    S[1] == S0
           + quicksum(buy_extra[i, 1] for i in I_extra),
    name="stock_init"
)
#     — En flujo semanal (t ≥ 2), agregamos compras de extra y retornos de mant.
for t in T[1:]:
    model.addConstr(
        S[t] == S[t-1]
               + quicksum(buy_extra[i, t] for i in I_extra)
               + quicksum(m[i, t-d] for i in I_WB if t-d > 0),
        name=f"stock_flow_{t}"
    )



# 4.8 Motor sigue en el mismo Avión que la semana Anterior
for i in I_WB:
    for p in P_WB:
        for t in T[1:]:  # desde semana 2 en adelante
            model.addConstr(
                a[i, p, t] >= a[i, p, t-1] - m[i, t],
                name=f"continuity{i}{p}{t}"
            )



# 4.9 Solo stock si lo compré antes de t
for i in I_extra:
    for t in T:
        model.addConstr(
            s[i, t]
            <= quicksum(buy_extra[i, τ] for τ in range(1, t+1)),
            name=f"no_stock_before_buy_{i}_{t}"
        )


# 4.10 Sólo puedes asignar un motor extra si ya lo compraste
for i in I_extra:
    for p in P_WB:
        for t in T:
            model.addConstr(
                a[i, p, t]
                <= quicksum(buy_extra[i, tau] for tau in range(1, t+1)),
                name=f"assign_only_if_bought_{i}_{p}_{t}"
            )


# 4.11. Si compro el motor extra i en la semana t, tiene que usarse ese mismo t
for i in I_extra:
    for t in T:
        model.addConstr(
            quicksum(a[i, p, t] for p in P_WB)
            >= buy_extra[i, t],
            name=f"use_bought_engine_{i}_{t}"
        )


# 4.12 Cada motor puede comprarse solo una vez
for i in I_extra:
    model.addConstr(
        quicksum(buy_extra[i,t] for t in T) <= 1,
        name=f"max_one_purchase_extra_{i}"
    )

# ─── (4.13) HEURÍSTICA en semanas pares ───────────────────────────────────────────────

# para t>1 en semanas impares:
for t in T[1:]:
    if t % 2 != 0:
        # 1) No iniciar nueva mantención
        for i in I_WB:
            model.addConstr(
                m[i, t] == 0,
                name=f"no_maint_odd_{i}_{t}"
            )
        # 2) No comprar motores extra
        for i in I_extra:
            model.addConstr(
                buy_extra[i, t] == 0,
                name=f"no_buy_odd_{i}_{t}"
            )
        # 3) No cambiar arrendamiento: lease[p,t] == lease[p,t-1]
        for p in P_WB:
            model.addConstr(
                ell[p, t] == ell[p, t-1],
                name=f"lease_const_odd_{p}_{t}"
            )
        # 4) No re-asignar motores: a[i,p,t] == a[i,p,t-1]
        for i in I_WB:
            for p in P_WB:
                model.addConstr(
                    a[i, p, t] == a[i, p, t-1],
                    name=f"assign_const_odd_{i}{p}{t}"
                )
        # 5) Stock permanece: s[i,t] == s[i,t-1]
        for i in I_WB:
            model.addConstr(
                s[i, t] == s[i, t-1],
                name=f"stock_const_odd_{i}_{t}"
            )

# ─── (4.14) Ciclo máximo estricto (HEURISTICA de semanas pares) ────────────────────────────────────────────────────────
for i in I_WB:
    for t in T:
        model.addConstr(
            y[i, t] <= C[i],
            name=f"cycle_limit_strict_no_over_{i}_{t}"
        )


# ─── (5) FUNCIÓN OBJETIVO Y OPTIMIZACION ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# --- Paso 5: Definir la función objetivo y optimizar ---

# 5.1. Función objetivo revisada: minimizar arrendos + compras por motor extra
model.setObjective(
    # coste de arrendar aviones
    quicksum(LeaseCost * ell[p, t]
             for p in P_WB for t in T)
    +
    # coste de comprar motores extra
    quicksum(BuyCost * buy_extra[i, t]
             for i in I_extra for t in T),
    GRB.MINIMIZE
)


# ─── (Heurística) WARM-START GREEDY ────────────────────────────────────────────────────────

# 1) Copiamos y0 para simular consumo de ciclos
cycles_greedy = y0.copy()

# 2) Reseteamos todos los Starts de binarias
for var in list(a.values()) + list(ell.values()) + list(buy_extra.values()):
    var.Start = 0

# 3) Asignación voraz semana a semana
for t in T:
    for p in P_WB:
        elegido = None
        # intentamos un motor propio
        for i in range(1, n_aviones+1):
            if cycles_greedy[i] + c[p] <= C[i]:
                elegido = i
                break
        if elegido:
            a[elegido, p, t].Start = 1
            cycles_greedy[elegido] += c[p]
        else:
            ell[p, t].Start = 1

# 4) Ajuste de Parámetros y Heurísticas de Gurobi
model.Params.Heuristics      = 0.2
model.Params.StartNodeLimit = 1_000_000
model.Params.MIPFocus   = 1    # más énfasis en heurísticas primales
model.Params.Cuts       = 0    # o prueba 2 para ver qué funciona mejor
model.Params.Threads    = 8    # tantas CPUs como tengas
model.Params.Presolve   = 2    # presolve agresivo
model.Params.TimeLimit  = 600  # opcional: corta si pasa 10 minutos



# ─── (6) OPTIMIZACION Y PRINTEAR RESULTADOS ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# 6.1. Ejecutar la optimización
model.optimize()

# 6.2. Obtener e imprimir resultados básicos
if model.status == GRB.OPTIMAL:
    print(f"Costo óptimo total: {model.objVal}")
    # Ejemplo: número total de motores arrendados y comprados
    total_leases = sum(ell[p, t].x for p in P_WB for t in T)
    total_buys = sum(buy_extra[i, t].x for i in I_extra for t in T)
    print(f"Total arrendos  : {int(total_leases)} semanas-motor")
    print(f"Total compras   : {int(total_buys)} motores comprados")
else:
    print("No se encontró solución óptima. Estado:", model.status)



# ─── (7) REGISTROS EN CSV ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# --- 1) CSV agrupado por avión ---
records_plane = []
lease_count    = {p: 0    for p in P_WB}   # contador de arrendos por avión

for t in T:
    for p in P_WB:
        # —––––– Aquí tienes que poner lo que ya hacías antes —–––––
        # 1) Extraer el motor asignado:
        motor = next((i for i in I_WB if a[i, p, t].X > 0.5), None)

        # 2) Ciclos acumulados de ese motor:
        cycles = y[motor, t].X if motor is not None else 0

        # 3) Umbral de ciclos según el avión p
        threshold = C[p]

        # 4) Lease tag (igual que antes):
        if ell[p, t].X > 0.5:
            lease_count[p] += 1
            lease_tag = f"lease_{lease_count[p]}"
        else:
            lease_tag = ""

        # 5) Bought: sólo si compraste ese motor extra en t
        if motor in I_extra and buy_extra[motor, t].X > 0.5:
            bought = "buy"
        else:
            bought = ""

        # 6) Swap (no lo usas todavía):
        swap = 0

        # 7) Sobreciclo?
        over = int(cycles > threshold)

        records_plane.append({
            'Semana':            t,
            'Avion_ID':          p,
            'Motor_asignado':    id2mat[motor] if motor else None,
            'Ciclos_acumulados': cycles,
            'Umbral_maximo':     threshold,
            'Leased':            lease_tag,
            'Bought':            bought,
            'Swap':              swap,
            'OverThreshold':     over
        })


df_plane = pd.DataFrame(records_plane)
df_plane.to_csv('processed_data/plane_weekly_status.csv', index=False)


# --- 2) CSV de reporte semanal ---
records_weekly = []
cum_cost = 0.0

for t in T:
    # costo acumulado
    cum_cost += sum(LeaseCost * ell[p, t].X for p in P_WB) \
            + sum(BuyCost * buy_extra[i, t].X for i in I_extra)

    # conteos
    n_mant = sum(int(r[i, t].X > 0.5) for i in I_WB)
    n_lease = sum(int(ell[p, t].X > 0.5) for p in P_WB)
    n_buy = sum(int(buy_extra[i, t].X > 0.5) for i in I_extra)
    n_swap  = sum(rec['Swap'] for rec in records_plane if rec['Semana'] == t)
    # Contar motores en sobreciclo esta semana
    n_over = sum(1 for rec in records_plane
                 if rec['Semana']==t and rec['OverThreshold'])

    # nuevo: motores en stock al inicio de la semana t
    n_stock = sum(int(s[i, t].X > 0.5) for i in I_WB)

    records_weekly.append({
        'Semana':                   t,
        'Num_Aviones':              len(P_WB),
        'Motores_en_mantenimiento': n_mant,
        'Motores_arrendados':       n_lease,
        'Motores_comprados':        n_buy,
        'Swaps_realizados':         n_swap,
        'Motores_en_stock':         n_stock,
        'Motores_sobreciclo':       n_over,        
        'Costo_acumulado':          cum_cost
    })

df_weekly_report = pd.DataFrame(records_weekly)
df_weekly_report.to_csv('processed_data/weekly_report.csv', index=False)
print(f"-> processed_data/weekly_report.csv ({len(df_weekly_report)} filas)")