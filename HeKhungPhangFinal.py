import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import symbols, Eq, solve, sympify, simplify

# === CẤU HÌNH TRANG ===
st.set_page_config(layout="wide")
st.title("Mô phỏng Hệ Khung Phẳng")

# === SIDEBAR: THÔNG SỐ VẬT LIỆU & HÌNH HỌC ===
st.sidebar.header("Thông số Vật liệu và Hình học")
E = float(st.sidebar.text_input("E (N/m²)", "2e11"))
v = float(st.sidebar.text_input("v", "0.3"))
A = float(st.sidebar.text_input("A (m²)", "0.01"))
I = float(st.sidebar.text_input("I (m⁴)", "0.00001"))
st.sidebar.markdown("---")
num_nodes    = st.sidebar.number_input("Số lượng Node",    min_value=2, step=1)
num_elements = st.sidebar.number_input("Số lượng phần tử", min_value=1, step=1)

# === NHẬP TOẠ ĐỘ & PHẦN TỬ ===
coords   = []
elements = []

st.subheader("Tọa độ các Node (x, y)")
for i in range(int(num_nodes)):
    c1, c2 = st.columns(2)
    x = c1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = c2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x, y))

st.subheader("Danh sách phần tử (i, j)")
for i in range(int(num_elements)):
    c1, c2 = st.columns(2)
    ni = c1.number_input(f"Node i - phần tử {i+1}", min_value=1, max_value=int(num_nodes), key=f"e{i}_start")
    nj = c2.number_input(f"Node j - phần tử {i+1}", min_value=1, max_value=int(num_nodes), key=f"e{i}_end")
    elements.append((int(ni)-1, int(nj)-1))

# === VẼ SƠ ĐỒ KHUNG ===
if st.button("Vẽ sơ đồ khung"):
    fig, ax = plt.subplots()
    for idx, (i, j) in enumerate(elements):
        x1, y1 = coords[i]; x2, y2 = coords[j]
        ax.plot([x1, x2], [y1, y2], 'bo-')
        ax.text((x1+x2)/2, (y1+y2)/2, str(idx+1), color="red")
    for idx, (x,y) in enumerate(coords):
        ax.text(x, y, str(idx+1), fontsize=10, color="green")
    ax.set_aspect("equal"); ax.set_title("Sơ đồ Hệ Khung")
    st.pyplot(fig)

# === TÍNH Ke & K_global ===
dof_per_node = 3
total_dof    = int(num_nodes) * dof_per_node
K_global     = np.zeros((total_dof, total_dof))

def element_dofs(n_i, n_j):
    return [
        n_i*3+0, n_i*3+1, n_i*3+2,
        n_j*3+0, n_j*3+1, n_j*3+2
    ]

if st.button("Tính Ke và K tổng thể"):
    st.subheader("Ma trận độ cứng phần tử (Ke)")
    Ke_list = []
    for idx, (i, j) in enumerate(elements):
        xi, yi = coords[i]; xj, yj = coords[j]
        L = sqrt((xj-xi)**2 + (yj-yi)**2)
        ang = radians(degrees(np.arctan2(yj-yi, xj-xi)))
        c = cos(ang); s = sin(ang)
        B = 12*I/(L**2)
        a11 = (A*c*c + B*s*s)/L
        a22 = (A*s*s + B*c*c)/L
        a12 = (A-B)*c*s/L
        a13 = -(B*L*s)/(2*L)
        a23 =  (B*L*c)/(2*L)
        a33 = 4*I/L; a36 = 2*I/L
        # xây Ke
        Ke = np.array([
            [ a11,  a12,  a13, -a11, -a12,  a13],
            [ a12,  a22,  a23, -a12, -a22,  a23],
            [ a13,  a23,  a33, -a13, -a23,  a36],
            [-a11, -a12, -a13,  a11,  a12, -a13],
            [-a12, -a22, -a23,  a12,  a22, -a23],
            [ a13,  a23,  a36, -a13, -a23,  a33],
        ])
        Ke = np.round(Ke, 5)
        Ke_list.append(Ke)
        # lắp vào K_global
        dofs = element_dofs(i, j)
        for m in range(6):
            for n in range(6):
                K_global[dofs[m], dofs[n]] += Ke[m,n]
        st.markdown(f"**Phần tử {idx+1} (L={L:.3f})**")
        st.text(Ke)

    st.subheader("Ma trận độ cứng tổng thể K")
    st.text(np.round(K_global, 5))

# === NHẬP Pe & LẮP RÁP P ===
st.subheader("Vector tải phần tử (Pe)")
Pe_list = []
for idx in range(int(num_elements)):
    st.markdown(f"Phần tử {idx+1}")
    cols = st.columns(6)
    pe = []
    for j in range(6):
        v = cols[j].text_input(f"Pe[{idx+1}][{j+1}]", "0", key=f"pe_{idx}_{j}")
        try:
            pe.append(float(sympify(v)))
        except:
            pe.append(0.0)
    Pe_list.append(pe)

P_global = None
if st.button("Lắp ráp vector tải toàn thể P từ Pe"):
    P_global = np.zeros((total_dof,1))
    for idx, (i,j) in enumerate(elements):
        dofs = element_dofs(i,j)
        for m, val in zip(dofs, Pe_list[idx]):
            P_global[m,0] += val
    st.subheader("Vector tải toàn thể P")
    st.text(np.round(P_global, 5))

# === Nhập q_known ===
st.subheader("Điều kiện biên: q = 0 tại các bậc tự do")
q_known = []
s = st.text_input("Nhập chỉ số q=0 (cách bởi dấu cách)", "")
if s:
    try:
        q_known = [int(x)-1 for x in s.split()]
    except:
        st.warning("Định dạng sai, ví dụ: 1 4 5")

# === Tính chuyển vị q ===
q_full = None
if st.button("Tính chuyển vị q"):
    if P_global is None:
        st.error("Bạn cần lắp ráp P trước")
    else:
        if len(q_known)>= total_dof:
            st.error("Không còn bậc tự do để giải")
        else:
            K_mod = np.delete(K_global, q_known, axis=0)
            K_mod = np.delete(K_mod, q_known, axis=1)
            P_mod = np.delete(P_global, q_known, axis=0)
            q_unk = np.linalg.solve(E*K_mod, P_mod)
            q_full = np.zeros((total_dof,1))
            cnt=0
            for i in range(total_dof):
                if i in q_known:
                    q_full[i,0]=0
                else:
                    q_full[i,0]=q_unk[cnt,0]; cnt+=1
            st.subheader("Vector chuyển vị q")
            st.text(np.round(q_full,5))

# === Tính phản lực liên kết R ===
if st.button("Tính phản lực liên kết R"):
    if q_full is None or P_global is None:
        st.error("Cần có q và P để tính phản lực")
    else:
        R = np.dot(K_global, q_full)
        st.subheader("Vector phản lực liên kết R")
        st.text(np.round(R,5))

# === HƯỚNG DẪN SỬ DỤNG ===
with st.expander("📘 Hướng dẫn sử dụng"):
    st.markdown("""
1. Nhập **E, v, A, I** trong thanh sidebar.
2. Chọn **số Node** và **số phần tử**.
3. Nhập **tọa độ Node** (x, y).
4. Định nghĩa **cấu trúc phần tử** (i → j).
5. Bấm **"Vẽ sơ đồ khung"** để kiểm tra.
6. Bấm **"Tính Ke và K tổng thể"**.
7. Nhập **Pe** cho mỗi phần tử.
8. Bấm **"Lắp ráp vector tải tổng thể P"**.
9. Nhập **các chỉ số q=0** (bậc tự do cố định).
10. Bấm **"Tính chuyển vị q"**.
11. Bấm **"Tính phản lực liên kết R"**.

* Lưu ý:
- Mỗi node có 3 bậc tự do (dof): dịch chuyển x, dịch chuyển y, quay.
- Chỉ số q bắt đầu từ 1.
    """)
