import pandas as pd
from gurobipy import Model, GRB, quicksum


# --- Paso 2 (con IDs numéricos): Leer datos y derivar C[id] ---

# 1. Cargar CSV de flota wide‐body
df_status_wb = pd.read_csv('Datos/Fleet_status_WB.csv')

# 2. Asignar IDs secuenciales según línea de archivo (1…n)
df_status_wb.insert(0, 'id', range(1, len(df_status_wb) + 1))

# 3. Mapeos matricula ↔ id
mat2id = dict(zip(df_status_wb['matricula'], df_status_wb['id']))
id2mat = {i: m for m, i in mat2id.items()}

# 4. Definir conjuntos de IDs
P_WB = df_status_wb['id'].tolist()   # IDs de aviones
I_WB = P_WB.copy()                   # IDs de motores (uno por avión)
T    = list(range(1, 30))            # horizonte de prueba: semanas 1…24

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

print("✅ Datos cargados.")



# --- Paso 3: Definir el modelo y las variables de decisión ---

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
# b[p,t]   = 1 si compramos un motor nuevo para el avión p en la semana t
b   = model.addVars(P_WB, T, vtype=GRB.BINARY, name="b")

# 3.6. NUEVA variable de swap: cambia de avión entre t-1 y t
# z = model.addVars(I_WB, T, vtype=GRB.BINARY, name="z")

# --- Paso 3.x: Asignación inicial en t=1 según el mismo id ---
# (pone cada motor i en el avión p=i durante la semana 1, y ninguno en los demás)
for p in P_WB:
    for i in I_WB:
        if i == p:
            model.addConstr(a[i, p, 1] == 1, name=f"init_assign_{i}_{p}")
        else:
            model.addConstr(a[i, p, 1] == 0, name=f"init_noassign_{i}_{p}")

# --- Paso 4: Agregar las restricciones ---

# 4.1 Estado exclusivo de cada motor: instalado (a), en mantenimiento (r) o en stock (s)
for i in I_WB:
    for t in T:
        model.addConstr(
            quicksum(a[i, p, t] for p in P_WB)  # suma de asignaciones
            + r[i, t]                           # más indicador de manutención
            + s[i, t]                           # más indicador de stock
            == 1,
            name=f"exclusive_{i}_{t}"
        )

# 4.2 Cobertura de cada avión: motor propio (a) o arrendado (ell) o comprado (b)
for p in P_WB:
    for t in T:
        model.addConstr(
            quicksum(a[i, p, t] for i in I_WB)  # si hay un motor i asignado a p
            + ell[p, t]                         # o lo arrendamos
            + b[p, t]                           # o compramos uno nuevo
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

# 4.4 Acumulación de ciclos: y[i,1] parte de y0, luego crece solo si a[i,p,t]=1
for i in I_WB:
    # semana 1
    model.addConstr(
        y[i, 1] == y0[i]
                  + quicksum(c[p] * a[i, p, 1] for p in P_WB),
        name=f"init_cycles_{i}"
    )
    # semanas 2…260
    for t in T[1:]:
        model.addConstr(
            y[i, t] == y[i, t-1]
                      + quicksum(c[p] * a[i, p, t] for p in P_WB),
            name=f"cycles_dyn_{i}_{t}"
        )

# 4.5 Límite dinámico de ciclos con big-M (corregido)

for i in I_WB:
    # Semana 1: usamos y0[i] en lugar de y[i,0]
    model.addConstr(
        y0[i]
        <= quicksum(C[p] * a[i, p, 1] for p in P_WB)
           + M * quicksum(m[i, tau] for tau in range(1, 2)),  # equivale a M*m[i,1]
        name=f"cycle_limit_{i}_1"
    )
    # Semanas 2…260: ahora sí y[i,t-1] existe
    for t in T[1:]:
        model.addConstr(
            y[i, t-1]
            <= quicksum(C[p] * a[i, p, t] for p in P_WB)
               + M * quicksum(m[i, tau] for tau in range(1, t+1)),
            name=f"cycle_limit_{i}_{t}"
        )


# 4.6 Capacidad de inicios de mantenimiento: a lo sumo M_max motores por semana
for t in T:
    model.addConstr(
        quicksum(m[i, t] for i in I_WB) <= M_max,
        name=f"capacity_{t}"
    )

# 4.7 Flujo de inventario de repuestos
#   - Semana 1: stock inicial + compras - arrendos
model.addConstr(
    S[1] == S0
           + quicksum(b[p, 1] for p in P_WB)
           - quicksum(ell[p, 1] for p in P_WB),
    name="stock_init"
)
#   - Semanas 2…260: stock anterior + compras + retornos de mantención - arrendos
for t in T[1:]:
    model.addConstr(
        S[t] == S[t-1]
               + quicksum(b[p, t] for p in P_WB)
               + quicksum(m[i, t-d] for i in I_WB if t-d > 0)
               - quicksum(ell[p, t] for p in P_WB),
        name=f"stock_flow_{t}"
    )

for i in I_WB:
    for p in P_WB:
        for t in T[1:]:  # desde semana 2 en adelante
            model.addConstr(
                a[i, p, t] >= a[i, p, t-1] - m[i, t],
                name=f"continuity{i}{p}{t}"
            )

"""
# 4.8 Definición de swap: z[i,t] ≥ |a[i,·,t]–a[i,·,t–1]|
for i in I_WB:
    for t in T[1:]:
        # cambio positivo
        model.addConstr(
            z[i, t] >= quicksum(a[i, p, t] - a[i, p, t-1] for p in P_WB),
            name=f"swap_pos_{i}_{t}"
        )
        # cambio negativo
        model.addConstr(
            z[i, t] >= quicksum(a[i, p, t-1] - a[i, p, t] for p in P_WB),
            name=f"swap_neg_{i}_{t}"
        )

# 4.9 Límite de 1 swap cada 2 semanas (ventana deslizante)
for t in T[1:]:
    model.addConstr(
        quicksum(z[i, tau] for i in I_WB for tau in [t-1, t]) <= 1,
        name=f"swap_limit_{t}"
    )
"""

"""
# 4.10 Prohibir compras al final si arrendar es más barato
H = max(T)
W_buy = (BuyCost // LeaseCost) + 1
for p in P_WB:
    for t in T:
        if t > H - W_buy:
            model.addConstr(
                b[p, t] == 0,
                name=f"no_buy_end_{p}_{t}"
            )
"""


# --- Paso 5: Definir la función objetivo y optimizar ---

# 5.1. Función objetivo: minimizar costo de arrendos y compras
model.setObjective(
    quicksum(LeaseCost * ell[p, t] + BuyCost * b[p, t]
             for p in P_WB for t in T),
    GRB.MINIMIZE
)

# 5.2. Ejecutar la optimización
model.optimize()

# 5.3. Obtener e imprimir resultados básicos
if model.status == GRB.OPTIMAL:
    print(f"Costo óptimo total: {model.objVal}")
    # Ejemplo: número total de motores arrendados y comprados
    total_leases = sum(ell[p, t].x for p in P_WB for t in T)
    total_buys   = sum(b[p, t].x   for p in P_WB for t in T)
    print(f"Total arrendos  : {int(total_leases)} semanas-motor")
    print(f"Total compras   : {int(total_buys)} motores comprados")
else:
    print("No se encontró solución óptima. Estado:", model.status)



import pandas as pd

# --- 1) CSV agrupado por avión ---
records_plane = []
prev_assignment = {p: None for p in P_WB}
lease_count    = {p: 0    for p in P_WB}   # contador de arrendos por avión

for t in T:
    for p in P_WB:
        motor  = next((i for i in I_WB if a[i, p, t].Xn > 0.5), None)
        cycles = y[motor, t].Xn if motor is not None else 0
        threshold = C[p]

        # Leased: lease_{n} si ℓ[p,t]=1, nada si es propio o comprado
        if ell[p, t].Xn > 0.5:
            lease_count[p] += 1
            lease_tag = f"lease_{lease_count[p]}"
        else:
            lease_tag = ""

        # Bought flag
        bought = "buy" if b[p, t].Xn > 0.5 else ""

        # Swap
        prev = prev_assignment[p]
        swap = int(prev is not None and motor is not None and motor != prev)
        prev_assignment[p] = motor

        records_plane.append({
            'Semana':            t,
            'Avion_ID':          p,
            'Motor_asignado':    id2mat[motor] if motor else None,
            'Ciclos_acumulados': cycles,
            'Umbral_maximo':     threshold,
            'Leased':            lease_tag,
            'Bought':            bought,
            'Swap':              swap
        })

df_plane = pd.DataFrame(records_plane)
df_plane.to_csv('processed_data/plane_weekly_status.csv', index=False)


# --- 2) CSV de reporte semanal ---
records_weekly = []
cum_cost = 0.0

for t in T:
    # costo acumulado
    cum_cost += sum(
        LeaseCost*ell[p, t].Xn + BuyCost*b[p, t].Xn
        for p in P_WB
    )
    records_weekly.append({
        'Semana':                    t,
        'Num_Aviones':               len(P_WB),
        'Motores_en_mantenimiento':  int(sum(r[i, t].Xn > 0.5 for i in I_WB)),
        'Motores_arrendados':        int(sum(ell[p, t].Xn > 0.5 for p in P_WB)),
        'Motores_comprados':         int(sum(b[p, t].Xn  > 0.5 for p in P_WB)),
        'Swaps_realizados':          int(sum(rec['Swap'] for rec in records_plane if rec['Semana']==t)),
        'Stock_motores_repuestos':   S[t].Xn,              # <-- nuevo
        'Costo_acumulado':           cum_cost
    })

df_weekly_report = pd.DataFrame(records_weekly)
df_weekly_report.to_csv('processed_data/weekly_report.csv', index=False)