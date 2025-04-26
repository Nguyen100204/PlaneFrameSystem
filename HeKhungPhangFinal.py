import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import sympify, simplify, Eq, solve, Symbol

# --- Cấu hình trang ---
st.set_page_config(layout="wide")
st.title("Mô phỏng Hệ Khung Phẳng")

# --- Sidebar: Thông số vật liệu & hình học ---
st.sidebar.header("Thông số Vật liệu & Hình học")
E = float(st.sidebar.text_input("E (N/m²)", "2e11"))
v = float(st.sidebar.text_input("v", "0.3"))
A = float(st.sidebar.text_input("A (m²)", "0.01"))
I = float(st.sidebar.text_input("I (m⁴)", "0.00001"))
st.sidebar.markdown("---")
num_nodes = st.sidebar.number_input("Số Node",    min_value=2, step=1)
num_elems = st.sidebar.number_input("Số phần tử", min_value=1, step=1)

# --- Khởi tạo session_state cho mọi kết quả ---
for key in [
    "coords", "elements", "index_elems",
    "Ke_list", "L_list", "K_global",
    "P_list", "P_elem", "Pn_expr",
    "q_known", "q_full", "R_reactions"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Nhập tọa độ Node ---
cols = st.columns(1)
st.subheader("Tọa độ Node (x, y)")
coords = []
for i in range(int(num_nodes)):
    c1, c2 = st.columns(2)
    x = c1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = c2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x,y))
st.session_state.coords = coords

# --- Nhập kết nối phần tử (i→j) ---
st.subheader("Cấu trúc phần tử (i → j)")
elements = []
for i in range(int(num_elems)):
    c1,c2 = st.columns(2)
    ni = c1.number_input(f"Node i - PT{i+1}", 1, int(num_nodes), key=f"ei{i}")
    nj = c2.number_input(f"Node j - PT{i+1}", 1, int(num_nodes), key=f"ej{i}")
    elements.append((ni-1, nj-1))
st.session_state.elements = elements

# --- Nhập DOF indices ---
st.subheader("DOF indices local→global mỗi PT")
index_elems = []
for i in range(int(num_elems)):
    s = st.text_input(f"PT{i+1} DOF indices", "1 2 3 4 5 6", key=f"idx{i}")
    try:
        idx = [int(x)-1 for x in s.split()]
        index_elems.append(idx if len(idx)==6 else [0]*6)
    except:
        index_elems.append([0]*6)
st.session_state.index_elems = index_elems

# --- Vẽ sơ đồ khung ---
if st.button("Vẽ sơ đồ khung"):
    fig, ax = plt.subplots()
    for k,(i,j) in enumerate(st.session_state.elements):
        x1,y1 = st.session_state.coords[i]; x2,y2 = st.session_state.coords[j]
        ax.plot([x1,x2],[y1,y2],'bo-')
        ax.text((x1+x2)/2,(y1+y2)/2,str(k+1),color='red')
    for k,(x,y) in enumerate(st.session_state.coords):
        ax.text(x,y,str(k+1),color='green')
    ax.set_aspect("equal"); ax.set_title("Sơ đồ Hệ Khung")
    st.pyplot(fig)

# --- Tính Ke & K_global ---
dof_per_node = 3
total_dof    = int(num_nodes)*dof_per_node

def compute_Ke(i,j):
    xi,yi = st.session_state.coords[i]; xj,yj = st.session_state.coords[j]
    L = sqrt((xj-xi)**2 + (yj-yi)**2)
    if L==0:
        return None,0.0
    alpha = radians(degrees(np.arctan2(yj-yi, xj-xi)))
    c,s = cos(alpha), sin(alpha)
    Bk = 12*I/(L**2)
    a11=(A*c*c + Bk*s*s)/L
    a22=(A*s*s + Bk*c*c)/L
    a12=(A-Bk)*c*s/L
    a13=-(Bk*L*s)/(2*L)
    a23=(Bk*L*c)/(2*L)
    a33, a36 = 4*I/L, 2*I/L
    Ke = np.array([
        [ a11, a12, a13, -a11,-a12, a13],
        [ a12, a22, a23, -a12,-a22, a23],
        [ a13, a23, a33, -a13,-a23, a36],
        [-a11,-a12,-a13,  a11, a12,-a13],
        [-a12,-a22,-a23,  a12, a22,-a23],
        [ a13, a23, a36, -a13,-a23, a33],
    ])
    return np.round(Ke,5), L

if st.button("Tính Ke & K tổng thể"):
    Ke_list, L_list = [], []
    K_global = np.zeros((total_dof, total_dof))
    for k,(i,j) in enumerate(st.session_state.elements):
        Ke,L = compute_Ke(i,j)
        Ke_list.append(Ke)
        L_list.append(L)
        if Ke is not None:
            dofs = st.session_state.index_elems[k]
            for m in range(6):
                for n in range(6):
                    K_global[dofs[m],dofs[n]] += Ke[m,n]
    st.session_state.Ke_list = Ke_list
    st.session_state.L_list = L_list
    st.session_state.K_global = K_global.copy()
    st.subheader("Ke từng phần tử")
    for k,Ke in enumerate(Ke_list,1):
        st.write(f"PT{k} (L={L_list[k-1]:.3f})"); st.write(Ke)
    st.subheader("Ma trận K tổng thể")
    st.write(np.round(K_global,5))

# --- Nhập Pe & lắp P_elem ---
st.subheader("Nhập tải phần tử (Pe)")
P_list = []
for k,(i,j) in enumerate(st.session_state.elements):
    st.markdown(f"PT{k+1}")
    c1,c2,c3 = st.columns(3)
    a = float(c1.number_input(f"a PT{k+1}",0.0,key=f"a{k}"))
    Type = c2.selectbox(f"Type PT{k+1}",["p0+","p0-","q0+","q0-","M+","M-","P+","P-"],key=f"t{k}")
    Q = float(c3.number_input(f"Q PT{k+1}",0.0,key=f"Q{k}"))
    Ke,L = st.session_state.Ke_list[k], st.session_state.L_list[k]
    if Ke is None:
        P_list.append([0]*6)
    else:
        alpha = radians(degrees(np.arctan2(st.session_state.coords[j][1]-st.session_state.coords[i][1],
                                            st.session_state.coords[j][0]-st.session_state.coords[i][0])))
        # (dùng công thức gốc giống trước)
        if Type=="p0+":
            P1,P2,P3 = Q*L*cos(alpha)/2, -Q*L*sin(alpha)/2, 0
            P4,P5,P6 = P1,P2,0
        elif Type=="p0-":
            P1,P2,P3 = -Q*L*cos(alpha)/2, Q*L*sin(alpha)/2,0
            P4,P5,P6 = P1,P2,0
        elif Type=="q0+":
            P_1,P_2,P_3,P_4 = Q*L/2, Q*L**2/12, Q*L/2, -Q*L**2/12
            P1,P2,P3 = -sin(alpha)*P_1, cos(alpha)*P_1, P_2
            P4,P5,P6 = -sin(alpha)*P_3, cos(alpha)*P_3, P_4
        elif Type=="q0-":
            P_1,P_2,P_3,P_4 = -Q*L/2, -Q*L**2/12, -Q*L/2, Q*L**2/12
            P1,P2,P3 = -sin(alpha)*P_1, cos(alpha)*P_1, P_2
            P4,P5,P6 = -sin(alpha)*P_3, cos(alpha)*P_3, P_4
        else:
            sign = 1 if Type in ("M+","P+") else -1
            if Type in ("M+","M-"):
                P_1 = sign*Q*((-6*a)/(L**2)+(6*a**2)/(L**3))
                P_2 = sign*Q*(1-((4*a)/L)+(3*a**2)/(L**2))
                P_3 = sign*Q*(((6*a)/(L**2))-((6*a**2)/(L**3)))
                P_4 = sign*Q*((-(2*a)/L)+(3*a**2)/(L**2))
            else:
                P_1 = sign*Q*(1-(3*a**2)/(L**2)+(2*a**3)/(L**3))
                P_2 = sign*Q*(a-(2*a**2)/L+(a**3)/(L**2))
                P_3 = sign*Q*((3*a**2)/(L**2)-(2*a**3)/(L**3))
                P_4 = sign*Q*((-(a**2)/L)+(a**3)/(L**2))
            P1,P2,P3 = -sin(alpha)*P_1, cos(alpha)*P_1, P_2
            P4,P5,P6 = -sin(alpha)*P_3, cos(alpha)*P_3, P_4
        P_list.append([P1,P2,P3,P4,P5,P6])
    st.write(P_list[-1])
st.session_state.P_list = P_list

if st.button("Lắp ráp Global Load Vector P từ Pe"):
    P_elem = np.zeros((total_dof,1))
    for k,pe in enumerate(st.session_state.P_list):
        dofs = st.session_state.index_elems[k]
        for m,val in zip(dofs,pe):
            P_elem[m,0] += val
    st.session_state.P_elem = P_elem
    st.subheader("Global Load Vector P")
    st.write(np.round(P_elem,5))

# --- Nhập nodal loads Pn biểu thức ---
st.subheader("Nhập tải tại DOF (Pn biểu thức)")
Pn_expr = []
for i in range(total_dof):
    s = st.text_input(f"Pn[{i+1}]", "0", key=f"Pn{i}")
    expr = str(simplify(sympify(s)))
    Pn_expr.append(expr)
    st.write(expr)
st.session_state.Pn_expr = Pn_expr

# --- Nhập q_known ngoài nút ---
q_str = st.text_input("Indices q=0 (vd: 1 4 5)", key="qfix")
q_known = [int(x)-1 for x in q_str.split()] if q_str else []
st.session_state.q_known = q_known

# --- Tính chuyển vị q (numeric) ---
if st.button("Tính chuyển vị q"):
    if st.session_state.P_elem is None:
        st.error("Chưa lắp ráp P_elem")
    elif st.session_state.K_global is None:
        st.error("Chưa tính K_global")
    else:
        Kg = st.session_state.K_global
        qk = st.session_state.q_known
        # giảm bậc tự do
        K_red = np.delete(Kg, qk, axis=0)
        K_red = np.delete(K_red, qk, axis=1)
        P_red = np.delete(st.session_state.P_elem, qk, axis=0)
        try:
            q_unknown = np.linalg.solve(E*K_red, P_red)
        except np.linalg.LinAlgError:
            st.error("K_mod singular; kiểm tra q_known")
        else:
            q_full = np.zeros((total_dof,1))
            cnt=0
            for i in range(total_dof):
                if i in qk:
                    q_full[i,0]=0
                else:
                    q_full[i,0]=q_unknown[cnt,0]; cnt+=1
            st.session_state.q_full = q_full
            st.subheader("q_full")
            st.write(np.round(q_full.T,5))

# --- Tính phản lực liên kết R (numeric) ---
if st.button("Tính phản lực liên kết"):
    if st.session_state.q_full is None:
        st.error("Chưa tính q_full")
    else:
        R = np.dot(st.session_state.K_global, E*st.session_state.q_full)
        st.session_state.R_reactions = R
        st.subheader("Phản lực liên kết R")
        st.write(np.round(R.T,5))

# --- Hướng dẫn ---
with st.expander("📘 Hướng dẫn"):
    st.markdown("""
1. Nhập E,v,A,I; Node; PT; DOF.
2. Vẽ khung.
3. Tính Ke & K.
4. Nhập Pe.
5. Lắp P_elem.
6. Nhập Pn_expr.
7. Nhập q_known.
8. Tính q.
9. Tính PLLK.
""")
