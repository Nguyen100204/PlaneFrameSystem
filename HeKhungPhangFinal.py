
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees, sqrt
from sympy import symbols, Eq, solve, sympify, simplify

st.set_page_config(layout="wide")
st.title("Mô phỏng Hệ Khung Phẳng")

st.sidebar.header("Thông số Vật liệu và Hình học")

E = float(st.sidebar.text_input("E (N/m²)", "2e11"))
v = float(st.sidebar.text_input("v", "0.3"))
A = float(st.sidebar.text_input("A (m²)", "0.01"))
I = float(st.sidebar.text_input("I (m⁴)", "0.00001"))

st.sidebar.markdown("---")

num_nodes = st.sidebar.number_input("Số lượng Node", min_value=2, step=1)
num_elements = st.sidebar.number_input("Số lượng phần tử", min_value=1, step=1)

coords = []
elements = []

st.subheader("Tọa độ các Node (x, y)")
for i in range(int(num_nodes)):
    col1, col2 = st.columns(2)
    x = col1.number_input(f"x{i+1}", key=f"x{i}", value=0.0)
    y = col2.number_input(f"y{i+1}", key=f"y{i}", value=0.0)
    coords.append((x, y))

st.subheader("Danh sách phần tử (i, j)")
for i in range(int(num_elements)):
    col1, col2 = st.columns(2)
    ni = col1.number_input(f"Node i - phần tử {i+1}", min_value=1, max_value=int(num_nodes), key=f"e{i}_start")
    nj = col2.number_input(f"Node j - phần tử {i+1}", min_value=1, max_value=int(num_nodes), key=f"e{i}_end")
    elements.append((int(ni)-1, int(nj)-1))

# Vẽ sơ đồ khung
if st.button("Vẽ sơ đồ khung"):
    fig, ax = plt.subplots()
    for i, (start, end) in enumerate(elements):
        x1, y1 = coords[start]
        x2, y2 = coords[end]
        ax.plot([x1, x2], [y1, y2], 'bo-', label=f"Phần tử {i+1}")
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{i+1}", color="red")

    for i, (x, y) in enumerate(coords):
        ax.text(x, y, f"{i+1}", fontsize=10, color="green")

    ax.set_aspect("equal")
    ax.set_title("Sơ đồ Hệ Khung")
    st.pyplot(fig)

# Tính Ke và K tổng thể
Ke_all = []
dof_per_node = 3
total_dof = int(num_nodes) * dof_per_node
K_global = np.zeros((total_dof, total_dof))

def element_dofs(node_i, node_j):
    return [
        node_i * 3 + 0, node_i * 3 + 1, node_i * 3 + 2,
        node_j * 3 + 0, node_j * 3 + 1, node_j * 3 + 2
    ]

if st.button("Tính K phần tử và K tổng thể"):
    st.subheader("Ma trận độ cứng phần tử và tổng thể")
    for idx, (i, j) in enumerate(elements):
        xi, yi = coords[i]
        xj, yj = coords[j]
        L = sqrt((xj - xi)**2 + (yj - yi)**2)
        angle = radians(degrees(np.arctan2((yj - yi), (xj - xi))))
        c = cos(angle)
        s = sin(angle)
        B = (12 * I) / (L ** 2)
        a11 = ((A * c**2 + B * s**2) / L)
        a22 = ((A * s**2 + B * c**2) / L)
        a33 = (4 * I / L)
        a12 = ((A - B) * c * s / L)
        a13 = (-(B * L * s) / 2 / L)
        a23 = ((B * L * c) / 2 / L)
        a36 = (2 * I / L)
        Ke = np.array([
            [a11, a12, a13, -a11, -a12, a13],
            [a12, a22, a23, -a12, -a22, a23],
            [a13, a23, a33, -a13, -a23, a36],
            [-a11, -a12, -a13, a11, a12, -a13],
            [-a12, -a22, -a23, a12, a22, -a23],
            [a13, a23, a36, -a13, -a23, a33]
        ])
        Ke = np.round(Ke, 5)
        Ke_all.append(Ke)
        dofs = element_dofs(i, j)
        for m in range(6):
            for n in range(6):
                K_global[dofs[m], dofs[n]] += Ke[m, n]

        st.markdown(f"**Phần tử {idx+1}** (L={L:.3f}):")
        st.text(Ke)

    st.subheader("Ma trận độ cứng tổng thể K")
    st.text(np.round(K_global, 5))

# Nhập tải trọng tại các Dof
st.subheader("Nhập tải trọng tại các bậc tự do (dof)")

load_vector = np.zeros((total_dof, 1))
for i in range(total_dof):
    val = st.text_input(f"P[{i+1}]", "0", key=f"P{i}")
    try:
        load_vector[i, 0] = float(sympify(val))
    except:
        st.warning(f"P[{i+1}] không hợp lệ, dùng giá trị 0")

# Nhập điều kiện q = 0
indices_str = st.text_input("Nhập chỉ số q = 0 (phản lực), cách nhau bởi dấu cách", "")
q_known = []
if indices_str:
    try:
        q_known = [int(i) - 1 for i in indices_str.strip().split()]
    except:
        st.warning("Lỗi định dạng chỉ số")

# Tính chuyển vị
if st.button("Tính chuyển vị q"):
    if len(q_known) >= total_dof:
        st.error("Tất cả q đều đã biết = 0, không thể giải")
    else:
        try:
            K_mod = np.delete(K_global, q_known, axis=0)
            K_mod = np.delete(K_mod, q_known, axis=1)
            P_mod = np.delete(load_vector, q_known, axis=0)
            q_unknown = np.linalg.solve(E * K_mod, P_mod)
            q_full = np.zeros((total_dof, 1))
            j = 0
            for i in range(total_dof):
                if i in q_known:
                    q_full[i] = 0
                else:
                    q_full[i] = q_unknown[j]
                    j += 1
            st.subheader("Vector chuyển vị q:")
            st.text(np.round(q_full, 5))
        except Exception as e:
            st.error(f"Lỗi khi giải hệ: {e}")

# ======= PHẢN LỰC LIÊN KẾT =======
if st.button("Tính phản lực liên kết"):
    try:
        P_goc = np.dot(K_global, q_full)
        st.subheader("Vector phản lực liên kết R:")
        st.text(np.round(P_goc, 5))

        st.markdown("### 🎯 Ý nghĩa:")
        st.markdown("- Đây là vector tải gốc do ma trận K và chuyển vị q tạo ra.")
        st.markdown("- Nếu có tải tại các Dof nào bị khống chế (q = 0), thì giá trị trong R sẽ là phản lực tại đó.")
    except:
        st.warning("Vui lòng tính chuyển vị q trước khi tính phản lực.")

# ======= HƯỚNG DẪN SỬ DỤNG =======
with st.expander("📘 Hướng Dẫn Sử Dụng"):
    st.markdown("""
### 🧰 Các bước sử dụng phần mềm:

1. **Nhập thông số vật liệu** ở thanh bên trái: E, v, A, I.
2. **Chọn số Node** và **số phần tử**.
3. **Nhập tọa độ các Node** theo thứ tự.
4. **Khai báo các phần tử** bằng cách chọn node i và node j.
5. Bấm **"Vẽ sơ đồ khung"** để kiểm tra sơ đồ hình học.
6. Bấm **"Tính K phần tử và K tổng thể"** để tạo ma trận độ cứng.
7. Nhập **vector tải trọng P** tại từng bậc tự do (dof).
8. Nhập **các chỉ số q = 0** (bậc tự do bị khống chế).
9. Bấm **"Tính chuyển vị q"** để giải hệ.
10. Cuối cùng, bấm **"Tính phản lực liên kết"** để xem phản lực tại các chỗ q = 0.

📌 Lưu ý:
- Mỗi node có 3 bậc tự do (dof): dịch chuyển x, dịch chuyển y, quay.
- Chỉ số q bắt đầu từ 1.
    """)
