import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import sympify, simplify, Symbol, Eq, solve

# --- Cấu hình ---
st.set_page_config(layout="wide")
st.title("Mô phỏng Hệ Khung Phẳng")

# --- Sidebar: vật liệu & thông số ---
st.sidebar.header("Thông số Vật liệu & Hình học")
E = float(st.sidebar.text_input("E (N/m²)", "2e11"))
v = float(st.sidebar.text_input("v", "0.3"))
A = float(st.sidebar.text_input("A (m²)", "0.01"))
I = float(st.sidebar.text_input("I (m⁴)", "0.00001"))
st.sidebar.markdown("---")
num_nodes = st.sidebar.number_input("Số Node",    min_value=2, step=1)
num_elems = st.sidebar.number_input("Số phần tử", min_value=1, step=1)

# --- Session state để lưu P_elem và q_full ---
if "P_elem" not in st.session_state:
    st.session_state.P_elem = None
if "q_full" not in st.session_state:
    st.session_state.q_full = None

# --- Nhập Node & phần tử, DOF indices như trước ---
coords, elements, index_elems = [], [], []
st.subheader("Tọa độ Node (x, y)")
for i in range(int(num_nodes)):
    c1,c2 = st.columns(2)
    x = c1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = c2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x,y))

st.subheader("Cấu trúc phần tử (i → j)")
for i in range(int(num_elems)):
    c1,c2 = st.columns(2)
    ni = c1.number_input(f"Node i - PT{i+1}", 1, int(num_nodes), key=f"ei{i}")
    nj = c2.number_input(f"Node j - PT{i+1}", 1, int(num_nodes), key=f"ej{i}")
    elements.append((int(ni)-1,int(nj)-1))

st.subheader("Chỉ số DOF local → global mỗi PT")
for i in range(int(num_elems)):
    s = st.text_input(f"PT{i+1} DOF indices", "1 2 3 4 5 6", key=f"idx{i}")
    try:
        idx = [int(x)-1 for x in s.split()]
        index_elems.append(idx if len(idx)==6 else [0]*6)
    except:
        index_elems.append([0]*6)

# --- Vẽ khung ---
if st.button("Vẽ sơ đồ khung"):
    fig, ax = plt.subplots()
    for k,(i,j) in enumerate(elements):
        x1,y1=coords[i]; x2,y2=coords[j]
        ax.plot([x1,x2],[y1,y2],'bo-')
        ax.text((x1+x2)/2,(y1+y2)/2,str(k+1),color='red')
    for k,(x,y) in enumerate(coords):
        ax.text(x,y,str(k+1),color='green')
    ax.set_aspect("equal"); ax.set_title("Sơ đồ Hệ Khung")
    st.pyplot(fig)

# --- Tính Ke & ma trận tổng thể K ---
dof_per_node = 3
total_dof    = int(num_nodes)*dof_per_node
K_global     = np.zeros((total_dof, total_dof))

def compute_Ke(i,j):
    xi,yi=coords[i]; xj,yj=coords[j]
    L = sqrt((xj-xi)**2+(yj-yi)**2)
    if L==0: return None, 0.0
    alpha = radians(degrees(np.arctan2(yj-yi, xj-xi)))
    c,s = cos(alpha), sin(alpha)
    Bk = 12*I/(L**2)
    a11=(A*c*c + Bk*s*s)/L
    a22=(A*s*s + Bk*c*c)/L
    a12=(A-Bk)*c*s/L
    a13=-(Bk*L*s)/(2*L)
    a23=(Bk*L*c)/(2*L)
    a33=4*I/L; a36=2*I/L
    Ke=np.array([
      [ a11, a12, a13, -a11,-a12, a13],
      [ a12, a22, a23, -a12,-a22, a23],
      [ a13, a23, a33, -a13,-a23, a36],
      [-a11,-a12,-a13,  a11, a12,-a13],
      [-a12,-a22,-a23,  a12, a22,-a23],
      [ a13, a23, a36, -a13,-a23, a33]
    ])
    return np.round(Ke,5), L

if st.button("Tính Ke & K tổng thể"):
    st.subheader("Ke từng phần tử")
    for k,(i,j) in enumerate(elements):
        Ke,L=compute_Ke(i,j)
        if Ke is None:
            st.error(f"PT{k+1}: L=0 (Node trùng nhau)")
            continue
        st.markdown(f"- PT{k+1} (L={L:.3f})")
        st.text(Ke)
        dofs=index_elems[k]
        for m in range(6):
            for n in range(6):
                K_global[dofs[m],dofs[n]]+=Ke[m,n]
    st.subheader("K tổng thể")
    st.text(np.round(K_global,5))

# --- Nhập Pe và lắp P_elem numeric ---
st.subheader("Nhập tải phần tử (Pe)")
P_list=[]
for k,(i,j) in enumerate(elements):
    st.markdown(f"Phần tử {k+1}")
    c1,c2,c3=st.columns(3)
    a=float(c1.number_input(f"a PT{k+1}",0.0,key=f"a{k}"))
    Type=c2.selectbox(f"Type PT{k+1}",["p0+","p0-","q0+","q0-","M+","M-","P+","P-"],key=f"t{k}")
    Q=float(c3.number_input(f"Q PT{k+1}",0.0,key=f"Q{k}"))
    Ke,L=compute_Ke(i,j)
    if Ke is None:
        P_list.append([0]*6)
    else:
        alpha = radians(degrees(np.arctan2(coords[j][1]-coords[i][1],coords[j][0]-coords[i][0])))
        if Type=="p0+":
            P1,P2,P3=(Q*L*cos(alpha)/2, -Q*L*sin(alpha)/2, 0)
            P4,P5,P6=(P1,P2,0)
        elif Type=="p0-":
            P1,P2,P3=(-Q*L*cos(alpha)/2, Q*L*sin(alpha)/2,0)
            P4,P5,P6=(P1,P2,0)
        elif Type=="q0+":
            P_1,P_2,P_3,P_4=(Q*L/2, Q*L**2/12, Q*L/2, -Q*L**2/12)
            P1,P2,P3=(-sin(alpha)*P_1, cos(alpha)*P_1, P_2)
            P4,P5,P6=(-sin(alpha)*P_3, cos(alpha)*P_3, P_4)
        elif Type=="q0-":
            P_1,P_2,P_3,P_4=(-Q*L/2, -Q*L**2/12, -Q*L/2, Q*L**2/12)
            P1,P2,P3=(-sin(alpha)*P_1, cos(alpha)*P_1, P_2)
            P4,P5,P6=(-sin(alpha)*P_3, cos(alpha)*P_3, P_4)
        else:
            sign = 1 if Type in ("M+","P+") else -1
            if Type in ("M+","M-"):
                P_1=sign*Q*((-6*a)/(L**2)+(6*a**2)/(L**3))
                P_2=sign*Q*(1-(4*a)/L+3*a**2/L**2)
                P_3=sign*Q*((6*a)/(L**2)-(6*a**2)/(L**3))
                P_4=sign*Q*(-2*a/L+3*a**2/L**2)
            else:
                P_1=sign*Q*(1-3*a**2/L**2+2*a**3/L**3)
                P_2=sign*Q*(a-2*a**2/L+a**3/L**2)
                P_3=sign*Q*(3*a**2/L**2-2*a**3/L**3)
                P_4=sign*Q*(-a**2/L+a**3/L**2)
            P1,P2,P3=(-sin(alpha)*P_1, cos(alpha)*P_1, P_2)
            P4,P5,P6=(-sin(alpha)*P_3, cos(alpha)*P_3, P_4)
        P_list.append([P1,P2,P3,P4,P5,P6])
    st.write(np.round(P_list[-1],5))

if st.button("Lắp ráp Global Load Vector P từ Pe"):
    P_elem=np.zeros((total_dof,1))
    for k,pe in enumerate(P_list):
        dofs=index_elems[k]
        for m,val in zip(dofs,pe):
            P_elem[m,0]+=val
    st.session_state.P_elem=P_elem
    st.subheader("Global Load Vector P")
    st.write(np.round(P_elem,5))

# --- Nhập tải nodal Pn biểu thức (chỉ để in) ---
st.subheader("Nhập tải nút (Pn biểu thức)")
Pn_expr=[]
for i in range(total_dof):
    s=st.text_input(f"Pn[{i+1}]", "0", key=f"Pn{i}")
    expr=str(simplify(sympify(s)))
    Pn_expr.append(expr)
    st.write(f"Pn[{i+1}]:", expr)

# --- Nhập điều kiện q_known ngoài nút ---
q_known_input=st.text_input("Indices q=0 (vd: 1 4 5)", key="qfix")
q_known=[int(x)-1 for x in q_known_input.split()] if q_known_input else []

# --- Tính chuyển vị q như def chuyenvi gốc ---
if st.button("Tính chuyển vị q"):
    if st.session_state.P_elem is None:
        st.error("Phải lắp ráp P từ Pe trước")
    else:
        n=total_dof
        K=K_global.copy()
        P=st.session_state.P_elem.flatten()
        q_dabiet=q_known
        try:
            K_red=np.delete(K, q_dabiet, axis=0)
            K_red=np.delete(K_red, q_dabiet, axis=1)
            P_red=np.delete(P, q_dabiet)
            q_unknown=np.linalg.solve(E*K_red, P_red)
            q_full=np.zeros((n,1))
            j=0
            for i in range(n):
                if i in q_dabiet:
                    q_full[i,0]=0
                else:
                    q_full[i,0]=q_unknown[j]; j+=1
            st.session_state.q_full=q_full
            st.subheader("Vector chuyển vị q")
            st.write(np.round(q_full.T,5))
        except Exception as e:
            st.error(f"Lỗi khi giải hệ: {e}")

# --- Tính phản lực liên kết như def Phanluclienket gốc ---
if st.button("Tính phản lực liên kết"):
    if "q_full" not in st.session_state:
        st.error("Phải tính q trước")
    else:
        try:
            q_full2=E*st.session_state.q_full
            P_goc=np.dot(K_global, q_full2).flatten()
            P_an=Pn_expr  # chuỗi biểu thức
            results=[]
            for i, expr in enumerate(P_an):
                try:
                    left=sympify(expr)
                    right=float(P_goc[i])
                    eq=Eq(left, right)
                    syms=list(eq.free_symbols)
                    if len(syms)==1:
                        var=syms[0]
                        sol=solve(eq, var)
                        if sol:
                            results.append(f"{var} = {round(float(sol[0]),5)}")
                except Exception:
                    continue
            if results:
                st.subheader("Phản lực liên kết")
                for r in results:
                    st.write(r)
            else:
                st.info("Không tìm thấy ẩn hoặc không giải được nghiệm.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- Hướng dẫn ---
with st.expander("📘 Hướng dẫn sử dụng"):
    st.markdown("""
1. Nhập E, v, A, I.
2. Chọn số Node & phần tử.
3. Nhập tọa độ Node.
4. Nhập phần tử và DOF indices.
5. Vẽ sơ đồ khung.
6. Tính Ke & K tổng thể.
7. Nhập a, Type, Q → tính Pe.
8. Lắp ráp Global Load Vector P từ Pe.
9. Nhập Pn biểu thức tại DOF.
10. Nhập indices q=0.
11. Tính chuyển vị q (numeric).
12. Tính phản lực liên kết (symbolic).
""")
