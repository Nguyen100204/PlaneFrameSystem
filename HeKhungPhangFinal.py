import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import sympify, simplify

# --- Cấu hình trang ---
st.set_page_config(layout="wide")
st.title("Mô phỏng Hệ Khung Phẳng")

# --- Sidebar: Vật liệu & Thông số ---
st.sidebar.header("Thông số Vật liệu & Hình học")
E = float(st.sidebar.text_input("E (N/m²)", "2e11"))
v = float(st.sidebar.text_input("v", "0.3"))
A = float(st.sidebar.text_input("A (m²)", "0.01"))
I = float(st.sidebar.text_input("I (m⁴)", "0.00001"))
st.sidebar.markdown("---")
num_nodes = st.sidebar.number_input("Số Node", min_value=2, step=1)
num_elems = st.sidebar.number_input("Số phần tử", min_value=1, step=1)

# --- Nhập tọa độ Node ---
coords = []
st.subheader("Tọa độ Node (x, y)")
for i in range(int(num_nodes)):
    c1, c2 = st.columns(2)
    x = c1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = c2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x, y))

# --- Nhập phần tử (i→j) ---
elements = []
st.subheader("Cấu trúc phần tử (i → j)")
for i in range(int(num_elems)):
    c1, c2 = st.columns(2)
    ni = c1.number_input(f"Node i - PT{i+1}", 1, int(num_nodes), key=f"ei{i}")
    nj = c2.number_input(f"Node j - PT{i+1}", 1, int(num_nodes), key=f"ej{i}")
    elements.append((int(ni)-1, int(nj)-1))

# --- Nhập chỉ số DOF cho từng phần tử ---
index_elems = []
st.subheader("Chỉ số DOF local → global cho mỗi PT")
for i in range(int(num_elems)):
    s = st.text_input(f"PT{i+1} DOF indices", "1 2 3 4 5 6", key=f"idx{i}")
    try:
        idx = [int(x)-1 for x in s.split()]
        if len(idx)==6:
            index_elems.append(idx)
        else:
            st.warning(f"PT{i+1}: cần 6 số, bạn nhập {len(idx)}")
            index_elems.append([0]*6)
    except:
        index_elems.append([0]*6)

# --- Vẽ sơ đồ khung ---
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

# --- Tính Ke & K_global ---
dof_per_node = 3
total_dof    = int(num_nodes)*dof_per_node
K_global     = np.zeros((total_dof, total_dof))

def compute_Ke(i,j):
    xi,yi = coords[i]; xj,yj = coords[j]
    L = sqrt((xj-xi)**2 + (yj-yi)**2)
    alpha = radians(degrees(np.arctan2(yj-yi, xj-xi)))
    c,s = cos(alpha), sin(alpha)
    B = 12*I/(L**2)
    a11=(A*c*c + B*s*s)/L
    a22=(A*s*s + B*c*c)/L
    a12=(A-B)*c*s/L
    a13=-(B*L*s)/(2*L)
    a23=(B*L*c)/(2*L)
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
    st.subheader("Ma trận độ cứng phần tử (Ke)")
    for k,(i,j) in enumerate(elements):
        Ke,L = compute_Ke(i,j)
        st.markdown(f"- PT{k+1} (L={L:.3f}):")
        st.text(Ke)
        # assemble
        dofs = index_elems[k]
        for m in range(6):
            for n in range(6):
                K_global[dofs[m], dofs[n]] += Ke[m,n]
    st.subheader("Ma trận độ cứng tổng thể K")
    st.text(np.round(K_global,5))

# --- Nhập tải phần tử Pe ---
st.subheader("Nhập tải phần tử (Pe)")
Pe_list = []
for k,(i,j) in enumerate(elements):
    st.markdown(f"Phần tử {k+1}")
    c1,c2,c3 = st.columns(3)
    a = float(c1.number_input(f"a PT{k+1}", value=0.0, key=f"a{k}"))
    Type = c2.selectbox(f"Loại PT{k+1}", ["p0+","p0-","q0+","q0-","M+","M-","P+","P-"], key=f"t{k}")
    Q = float(c3.number_input(f"Q PT{k+1}", value=0.0, key=f"Q{k}"))
    Ke,L = compute_Ke(i,j)
    alpha = radians(degrees(np.arctan2(coords[j][1]-coords[i][1], coords[j][0]-coords[i][0])))
    # tính Pe theo code gốc
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
    elif Type in ("M+","M-","P+","P-"):
        sign = 1 if Type in ("M+","P+") else -1
        # ví dụ với M: dùng code gốc
        if Type in ("M+","M-"):
            P_1 = sign*Q*((-6*a)/(L**2)+(6*(a**2))/(L**3))
            P_2 = sign*Q*(1-((4*a)/L)+(3*(a**2)/(L**2)))
            P_3 = sign*Q*(((6*a)/(L**2))-((6*(a**2))/(L**3)))
            P_4 = sign*Q*((-(2*a)/L)+(3*(a**2)/(L**2)))
        else:
            P_1 = sign*Q*(1-(3*(a**2)/(L**2))+(2*(a**3)/(L**3)))
            P_2 = sign*Q*(a-(2*(a**2)/L)+((a**3)/(L**2)))
            P_3 = sign*Q*((3*(a**2)/(L**2))-(2*(a**3)/(L**3)))
            P_4 = sign*Q*((-(a**2)/L)+((a**3)/(L**2)))
        P1=-sin(alpha)*P_1; P2=cos(alpha)*P_1; P3=P_2
        P4=-sin(alpha)*P_3; P5=cos(alpha)*P_3; P6=P_4
    else:
        P1=P2=P3=P4=P5=P6=0
    Pe_list.append([P1,P2,P3,P4,P5,P6])
    st.text(np.round([P1,P2,P3,P4,P5,P6],5))

# --- Nhập Pn (tải tại DOF) ---
st.subheader("Nhập tải tại DOF (Pn)")
B_expr = []
for i in range(total_dof):
    s = st.text_input(f"Pₙ[{i+1}]", "0", key=f"B{i}")
    try:
        B_expr.append(str(simplify(sympify(s))))
    except:
        B_expr.append("0")

# --- Tính Pn và P_global (Pe + B) ---
Pn_list = None
P_global = None
if st.button("Tính Global Load Vector P"):
    # lắp Pe
    P_elem = np.zeros((total_dof,1))
    for k,pe in enumerate(Pe_list):
        dofs = index_elems[k]
        for m,val in zip(dofs,pe):
            P_elem[m,0] += val
    # P = Pe + Pn
    Pn_list = []
    P_global = np.zeros((total_dof,1))
    for i in range(total_dof):
        num_pe = P_elem[i,0]
        expr = sympify(B_expr[i])
        total = simplify(expr + num_pe)
        Pn_list.append(str(total))
        P_global[i,0] = float(np.round(float(simplify(expr + num_pe)),5))
    st.subheader("Vector tải P (Pe + Pₙ)")
    for idx,val in enumerate(Pn_list,1):
        st.write(f"P[{idx}] = {val}")

# --- Nhập q_known và giải chuyển vị ---
q_full = None
q_known = []
st.subheader("Điều kiện q = 0 (các DOF cố định)")
s = st.text_input("Các index q=0 (vd: 1 4 5)", "")
if s:
    try:
        q_known = [int(x)-1 for x in s.split()]
    except:
        st.warning("Định dạng sai")

if st.button("Tính chuyển vị q"):
    if P_global is None:
        st.error("Phải tính Pₙ trước")
    else:
        Km = np.delete(K_global, q_known, axis=0)
        Km = np.delete(Km, q_known, axis=1)
        Pm = np.delete(P_global, q_known, axis=0)
        sol = np.linalg.solve(E*Km, Pm)
        q_full = np.zeros((total_dof,1))
        cnt = 0
        for i in range(total_dof):
            if i in q_known:
                q_full[i,0] = 0
            else:
                q_full[i,0] = sol[cnt,0]; cnt += 1
        st.subheader("Vector chuyển vị q")
        st.text(np.round(q_full,5))

# --- Tính phản lực liên kết R ---
if st.button("Tính phản lực liên kết R"):
    if q_full is None:
        st.error("Phải tính q trước")
    else:
        R = np.dot(K_global, q_full)
        st.subheader("Vector phản lực liên kết R")
        st.text(np.round(R,5))

# --- Hướng dẫn sử dụng ---
with st.expander("📘 Hướng dẫn"):
    st.markdown("""
1. Nhập E, v, A, I.
2. Chọn số Node & phần tử.
3. Nhập tọa độ Node.
4. Định nghĩa phần tử (i → j).
5. Nhập mapping DOF cho mỗi phần tử.
6. Vẽ sơ đồ khung.
7. Tính Ke & K tổng thể.
8. Nhập a, Type, Q → tính Pe.
9. Nhập B tại mỗi DOF → tính Pₙ (Pe + B).
10. Nhập q=0 → tính chuyển vị q.
11. Tính phản lực liên kết R.
""")
