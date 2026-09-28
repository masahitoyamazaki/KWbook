"""
書籍『対称性をめぐって —圏論的対称性への誘い—』（山崎雅人 著）付録B に付随するコード
サポートページ: https://github.com/masahitoyamazaki/KWbook

横磁場イジング模型の厳密対角化（付録B）

    H_Ising(g) = -g^{-1} sum_j Z_j Z_{j+1} - g sum_j X_j      （周期的境界条件）

必要なもの: numpy, scipy, matplotlib
実行: python ising_ed.py      （このファイルと同じ場所の figures/ に図を出力する）
説明つきの版は同じフォルダの ising_ed.ipynb

Z2 対称性 eta = prod_j X_j の固有値ごとに対角化するため，
全サイトにアダマール変換をかけた基底（X と Z を入れ替えた基底）で計算する．
この基底では eta = prod_j Z_j が対角になり，ハミルトニアンは
    H = -g^{-1} sum_j X_j X_{j+1} - g sum_j Z_j
となる．
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def hamiltonian_sector(L, g, eta):
    """eta = +1 または -1 のセクターに制限したハミルトニアン（疎行列）を返す．"""
    # 基底: 0,...,2^L-1 の整数 n を L ビットのスピン配位とみなす（ビット i が si）
    states = np.arange(2 ** L)
    popcount = np.array([bin(n).count("1") for n in states])
    # prod_j Z_j = (-1)^{1 の個数}
    sector = states[(-1) ** popcount == eta]
    index = {n: k for k, n in enumerate(sector)}
    dim = len(sector)

    rows, cols, vals = [], [], []
    for k, n in enumerate(sector):
        # 対角成分: -g sum_j Z_j,  Z_j |s> = (-1)^{s_j} |s>
        diag = -g * sum((-1) ** ((n >> j) & 1) for j in range(L))
        rows.append(k); cols.append(k); vals.append(diag)
        # 非対角成分: -g^{-1} X_j X_{j+1} はビット j と j+1 を反転する
        for j in range(L):
            m = n ^ (1 << j) ^ (1 << ((j + 1) % L))
            rows.append(index[m]); cols.append(k); vals.append(-1.0 / g)
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def lowest_levels(L, g, eta, k=4):
    """セクター eta の低い方から k 個のエネルギー固有値．"""
    H = hamiltonian_sector(L, g, eta)
    if H.shape[0] <= 64:
        return np.sort(np.linalg.eigvalsh(H.toarray()))[:k]
    return np.sort(spla.eigsh(H, k=k, which="SA", return_eigenvectors=False))


def exact_ground_energy(L, g):
    """付録Cのジョルダン＝ウィグナー変換による eta=+1 セクターの基底エネルギー．"""
    ks = 2 * np.pi * (np.arange(L) + 0.5) / L          # 反周期的境界条件の運動量
    eps = 2 * np.sqrt(g ** 2 + g ** -2 - 2 * np.cos(ks))
    return -0.5 * eps.sum()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    # --- 1. クラマース＝ワニエ双対性と厳密解のチェック ---
    L = 10
    print("KW duality check: eta=+1 spectrum of H(g) vs H(1/g)")
    for g in [0.5, 0.8, 1.5]:
        e1 = lowest_levels(L, g, +1, k=6)
        e2 = lowest_levels(L, 1 / g, +1, k=6)
        print(f"  g={g}:  max|E(g)-E(1/g)| = {np.max(np.abs(e1 - e2)):.2e}")
    print("Jordan-Wigner check: ground energy in eta=+1 sector")
    for g in [0.5, 1.0, 2.0]:
        print(f"  g={g}:  ED = {lowest_levels(L, g, +1, 1)[0]:.10f},"
              f"  exact = {exact_ground_energy(L, g):.10f}")

    # --- 2. 低エネルギースペクトル（基底エネルギーからの差）---
    L = 12
    gs = np.linspace(0.3, 2.5, 60)
    plus = np.array([lowest_levels(L, g, +1, k=4) for g in gs])
    minus = np.array([lowest_levels(L, g, -1, k=4) for g in gs])
    e0 = np.minimum(plus[:, 0], minus[:, 0])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for n in range(4):
        ax.plot(gs, plus[:, n] - e0, "-", color="C0", lw=1.4,
                label=r"$\eta=+1$" if n == 0 else None)
        ax.plot(gs, minus[:, n] - e0, "--", color="C3", lw=1.4,
                label=r"$\eta=-1$" if n == 0 else None)
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$E_n - E_1$")
    ax.set_ylim(0, 8)
    ax.set_title(rf"Transverse-field Ising, $L={L}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ed_spectrum.pdf"))

    # --- 3. ギャップのサイズ依存性（eta=+1 セクター内の第一励起との差）---
    fig, ax = plt.subplots(figsize=(5, 3.6))
    gs = np.linspace(0.3, 2.5, 45)
    for L in [6, 8, 10, 12, 14]:
        gap = [np.diff(lowest_levels(L, g, +1, k=2))[0] for g in gs]
        ax.plot(gs, gap, label=rf"$L={L}$")
    ax.plot(gs, 4 * np.abs(gs - 1 / gs), "k:", lw=1, label=r"$4|g-g^{-1}|$")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"gap in $\eta=+1$ sector")
    ax.set_ylim(0, 8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ed_gap.pdf"))
    print("figures written to", os.path.abspath(outdir))
