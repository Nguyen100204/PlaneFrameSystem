import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import sympify, simplify, Symbol, Matrix, solve

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

# --- Chuẩn bị session_state cho P_global_expr ---
if "P_global_expr" not in st.session_state:
    st.session_state.P_global_expr = []

# --- Nhập tọa độ Node ---
coords = []
st.subheader("Tọa độ Node (x, y)")
for i in range(int(num_nodes)):
    c1, c2 = st.columns(2)
    x = c1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = c2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x, y))

# --- Nhập kết nối phần tử (i → j) ---
elements = []
st.subheader("Cấu trúc phần tử (i → j)")
for i in range(int(num_elems)):
    c1, c2 = st.columns(2)
    ni = c1.number_input(f"Node i - PT{i+1}", 1, int(num_nodes), key=f"ei{i}")
    nj = c2.number_input(f"Node j - PT{i+1}", 1, int(num_nodes), key=f"ej{i}")
    elements.append((int(ni)-1, int(nj)-1))

# --- Nhập DOF indices cho mỗi phần tử ---
index_elems = []
st.subheader("Chỉ số DOF local → global cho mỗi PT")
for i in range(int(num_elems)):
    s = st.text_input(f"PT{i+1} DOF indices", "1 2 3 4 5 6", key=f"idx{i}")
    try:
        idx = [int(x)-1 for x in s.split()]
        index_elems.append(idx if len(idx)==6 else [0]*6)
    except:
        index_elems.append([0]*6)

# --- Nút vẽ sơ đồ khung ---
if st.button("Vẽ sơ đồ khung"):
    fig, ax = plt.subplots()
    for k,(i,j) in enumerate(elements):
        x1,y1 = coords[i]; x2,y2 = coords[j]
        ax.plot([x1,x2],[y1,y2],'bo-')
        ax.text((x1+x2)/2,(y1+y2)/2,str(k+1), color='red')
    for k,(x,y) in enumerate(coords):
        ax.text(x,y,str(k+1), color='green')
    ax.set_aspect("equal"); ax.set_title("Sơ đồ Hệ Khung")
    st.pyplot(fig)

# --- Tính Ke & lắp K_global ---
dof_per_node = 3
total_dof    = int(num_nodes)*dof_per_node
K_global     = np.zeros((total_dof, total_dof))

def compute_Ke(i,j):
    xi,yi = coords[i]; xj,yj = coords[j]
    L = sqrt((xj-xi)**2 + (yj-yi)**2)
    if L == 0:
        return None, 0.0
    alpha = radians(degrees(np.arctan2(yj-yi, xj-xi)))
    c, s = cos(alpha), sin(alpha)
    Bk = 12*I/(L**2)
    a11=(A*c*c + Bk*s*s)/L
    a22=(A*s*s + Bk*c*c)/L
    a12=(A-Bk)*c*s/L
    a13=-(Bk*L*s)/(2*L)
    a23=(Bk*L*c)/(2*L)
    a33=4*I/L; a36=2*I/L
    Ke = np.array([
        [ a11,  a12,  a13, -a11, -a12,  a13],
        [ a12,  a22,  a23, -a12, -a22,  a23],
        [ a13,  a23,  a33, -a13, -a23,  a36],
        [-a11, -a12, -a13,  a11,  a12, -a13],
        [-a12, -a22, -a23,  a12,  a22, -a23],
        [ a13,  a23,  a36, -a13, -a23,  a33],
    ])
    return np.round(Ke,5), L

if st.button("Tính Ke & K tổng thể"):
    st.subheader("Ke từng phần tử")
    for k,(i,j) in enumerate(elements):
        Ke, L = compute_Ke(i,j)
        if Ke is None:
            st.error(f"PT{k+1}: hai Node trùng tọa độ (L=0)")
            continue
        st.markdown(f"- PT{k+1} (L={L:.3f}):")
        st.text(Ke)
        dofs = index_elems[k]
        for m in range(6):
            for n in range(6):
                K_global[dofs[m], dofs[n]] += Ke[m,n]
    st.subheader("Ma trận độ cứng tổng thể K")
    st.text(np.round(K_global,5))

# --- Nhập tải phần tử Pe ---
st.subheader("Nhập tải phần tử (Pe)")
P_list = []
for k,(i,j) in enumerate(elements):
    st.markdown(f"Phần tử {k+1}")
    c1,c2,c3 = st.columns(3)
    a = float(c1.number_input(f"a PT{k+1}", value=0.0, key=f"a{k}"))
    Type = c2.selectbox(f"Loại PT{k+1}", ["p0+","p0-","q0+","q0-","M+","M-","P+","P-"], key=f"t{k}")
    Q = float(c3.number_input(f"Q PT{k+1}", value=0.0, key=f"Q{k}"))
    Ke,L = compute_Ke(i,j)
    if Ke is None:
        P_list.append([0]*6)
    else:
        alpha = radians(degrees(np.arctan2(coords[j][1]-coords[i][1], coords[j][0]-coords[i][0])))
        # (các công thức Pe nguyên bản)
        if Type=="p0+":
            P1=(Q*L*cos(alpha))/2; P2=-(Q*L*sin(alpha))/2; P3=0
            P4=P1; P5=P2; P6=0
        elif Type=="p0-":
            P1=-(Q*L*cos(alpha))/2; P2=(Q*L*sin(alpha))/2; P3=0
            P4=P1; P5=P2; P6=0
        elif Type=="q0+":
            P_1=(Q*L)/2; P_2=(Q*L**2)/12; P_3=(Q*L)/2; P_4=-(Q*L**2)/12
            P1=-sin(alpha)*P_1; P2=cos(alpha)*P_1; P3=P_2
            P4=-sin(alpha)*P_3; P5=cos(alpha)*P_3; P6=P_4
        elif Type=="q0-":
            P_1=-(Q*L)/2; P_2=-(Q*L**2)/12; P_3=-(Q*L)/2; P_4=(Q*L**2)/12
            P1=-sin(alpha)*P_1; P2=cos(alpha)*P_1; P3=P_2
            P4=-sin(alpha)*P_3; P5=cos(alpha)*P_3; P6=P_4
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
                P_4 = sign*Q*((-(a**2)/L)+((a**3)/(L**2)))
            P1 = -sin(alpha)*P_1; P2=cos(alpha)*P_1; P3=P_2
            P4 = -sin(alpha)*P_3; P5=cos(alpha)*P_3; P6=P_4
        P_list.append([P1,P2,P3,P4,P5,P6])
    st.text(np.round(P_list[-1],5))

# --- Nhập tải tại DOF Pn biểu thức ---
st.subheader("Nhập tải tại DOF (Pn biểu thức)")
Pn_expr = []
for i in range(total_dof):
    s = st.text_input(f"Pn[{i+1}]", "0", key=f"Pn{i}")
    try:
        Pn_expr.append(str(simplify(sympify(s))))
    except:
        Pn_expr.append("0")

# --- Tính Global Load Vector P và lưu vào session_state ---
if st.button("Tính Global Load Vector P"):
    P_global_expr = []
    # assemble Pe numeric
    P_elem = np.zeros((total_dof,1))
    for k, pe in enumerate(P_list):
        dofs = index_elems[k]
        for m,val in zip(dofs,pe):
            P_elem[m,0] += val
    # cộng với Pn_expr, giữ ẩn
    st.subheader("Vector tải toàn thể P (biểu thức)")
    for i in range(total_dof):
        pe_i   = P_elem[i,0]
        expr_i = sympify(Pn_expr[i])
        total_i = simplify(expr_i + pe_i)
        P_global_expr.append(total_i)
        st.write(f"P[{i+1}] = {expr_i} + {pe_i} = **{total_i}**")
    st.session_state.P_global_expr = P_global_expr

# --- Nhập q_known (bên ngoài nút) ---
q_known_str = st.text_input("Indices q=0 (vd: 1 4 5)", key="qfix")
q_known = []
if q_known_str:
    try:
        q_known = [int(x)-1 for x in q_known_str.split()]
    except:
        st.warning("Định dạng sai q=0")

# --- Tính chuyển vị q symbolically ---
if st.button("Tính chuyển vị q"):
    if not st.session_state.P_global_expr:
        st.error("Phải tính Global Load Vector trước")
    else:
        total_dof = len(st.session_state.P_global_expr)
        q = Matrix([Symbol(f"q{i+1}") for i in range(total_dof)])
        K_sym = Matrix(E * K_global)
        P_sym = Matrix(st.session_state.P_global_expr)
        balance = K_sym * q - P_sym
        unknowns = [q[i] for i in range(total_dof) if i not in q_known]
        eqs      = [balance[i] for i in range(total_dof) if i not in q_known]
        sol_list = solve(eqs, unknowns, dict=True)
        if sol_list:
            sol = sol_list[0]
            q_full_expr = []
            for i in range(total_dof):
                if i in q_known:
                    q_full_expr.append(0)
                else:
                    q_full_expr.append(sol[q[i]])
            st.subheader("Vector chuyển vị q (biểu thức)")
            for i,val in enumerate(q_full_expr,1):
                st.write(f"q[{i}] = {val}")
            st.session_state.q_full_expr = q_full_expr
        else:
            st.error("Không giải được q symbolically")

# --- Tính phản lực liên kết R symbolically ---
if st.button("Tính phản lực liên kết R"):
    if "q_full_expr" not in st.session_state:
        st.error("Phải tính q trước")
    else:
        q_vec = Matrix(st.session_state.q_full_expr)
        K_sym = Matrix(E * K_global)
        R = K_sym * q_vec
        st.subheader("Vector phản lực liên kết R")
        for i,val in enumerate(R,1):
            st.write(f"R[{i}] = {val}")

# --- Hướng dẫn sử dụng ---
with st.expander("📘 Hướng dẫn sử dụng"):
    st.markdown("""
1. Nhập E, v, A, I.
2. Chọn số Node & phần tử.
3. Nhập tọa độ Node.
4. Nhập cấu trúc phần tử (i→j).
5. Nhập DOF indices cho mỗi phần tử.
6. Vẽ sơ đồ khung.
7. Tính Ke & K tổng thể.
8. Nhập a, Type, Q → tính Pe.
9. Nhập Pn biểu thức tại DOF.
10. Bấm **Tính Global Load Vector P**.
11. Nhập **Indices q=0** bên ngoài.
12. Bấm **Tính chuyển vị q**.
13. Bấm **Tính phản lực liên kết R**.
""")
