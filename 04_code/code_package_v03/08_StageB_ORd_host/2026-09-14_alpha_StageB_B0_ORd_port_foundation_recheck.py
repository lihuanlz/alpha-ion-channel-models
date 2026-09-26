# -*- coding: utf-8 -*-
"""
2026-09-14 · α模型 Stage B · B0 地基复核
========================================
Python 逐方程移植官方 newordherg_qNet.c（IKr-dynamic ORd，49 状态 + qNet 积分器），
control 无药，CL=1000 ms 起搏。参数/初值运行时直接读官方 txt（不硬编码）。

判线（预注册 §四 + §十一 v2 修订，跑前钉死）：
  1. 稳态化：末 10 拍 APD90 极差 < 0.5 ms；
  2. control 形态：末拍 APD90 ∈ [250,330] ms、qNet ∈ [0.05,0.12] µC/µF、
     静息电位 ∈ [−95,−85] mV、APA ∈ [90,130] mV；
  3. qNet 双口径自洽登记：整周期 = 状态49差分×1e-3；AP 窗内 = 六电流窗内积分×1e-3（末拍）；
  4.（v3 重构）Euler 拍1 双步长收敛签名：e(0.005)/e(0.00125) ∈ [2,6]；
     细网格 max|Δv| < 5 mV 且 argmax ≤ 10 ms；Euler-LSODA 拍1 ΔAPD90 < 1 ms；
     LSODA(1e-7) vs LSODA(1e-9) 拍1 max|Δv| < 0.5 mV。

纪律：本侧仅 ast.parse + SMOKE=1 冒烟（100 拍）；正式跑（1000 拍）用户 Spyder：
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_StageB_B0_ORd移植地基复核.py' --wdir
输出：本脚本同目录 _结果.json/.png（冒烟带 _冒烟 后缀）。
"""

import os
import sys
import json
import math
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(BASE, "CiPA官方锚")
PARS_FILE = os.path.join(ANCHOR, "newordherg_pars.txt")
STATES_FILE = os.path.join(ANCHOR, "newordherg_states_CL2000.txt")

SMOKE = os.environ.get("SMOKE", "0") == "1"
NBEATS = int(os.environ.get("SMOKE_BEATS", "100")) if SMOKE else int(os.environ.get("NBEATS", "1000"))
CL = 1000.0          # ms
DT_REC = 0.1         # ms 记录网格
AMP = -80.0          # 刺激幅值（从 pars 读，此处仅注释锚）
DUR = 0.5            # ms 刺激时长

PARS_NAMES = ["celltype", "GKrfc", "GNaLfc", "GNafc", "GKsfc", "GK1fc", "PCafc", "Gtofc",
              "A1", "B1", "q1", "A2", "B2", "q2", "A3", "B3", "q3", "A4", "B4", "q4",
              "A11", "B11", "q11", "A21", "B21", "q21", "A31", "B31", "q31", "A41", "B41", "q41",
              "A51", "B51", "q51", "A52", "B52", "q52", "A53", "B53", "q53",
              "A61", "B61", "q61", "A62", "B62", "q62", "A63", "B63", "q63",
              "Kmax", "Ku", "n", "halfmax", "Kt", "Vhalf", "T", "ko", "amp"]

STATE_NAMES = ["v", "nai", "nass", "ki", "kss", "cai", "cass", "cansr", "cajsr",
               "m", "hf", "hs", "j", "hsp", "jp", "mL", "hL", "hLp",
               "a", "iF", "iS", "ap", "iFp", "iSp",
               "d", "ff", "fs", "fcaf", "fcas", "jca", "nca", "ffp", "fcafp",
               "xs1", "xs2", "xk1", "Jrelnp", "Jrelp", "CaMKt",
               "IC1", "IC2", "C1", "C2", "O", "IO", "IObound", "Obound", "Cbound", "D"]


def load_named_values(path, expected):
    names, vals = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            names.append(parts[0])
            vals.append(float(parts[1]))
    if names != expected:
        raise RuntimeError(f"{os.path.basename(path)} 名称/顺序与官方锚不符: {names[:6]}...")
    return vals


# ----------------------------------------------------------------------------
# 模型本体：逐方程对照 newordherg_qNet.c derivs()
# ----------------------------------------------------------------------------
def build_model(P):
    (celltype, GKrfc, GNaLfc, GNafc, GKsfc, GK1fc, PCafc, Gtofc,
     A1, B1, q1, A2, B2, q2, A3, B3, q3, A4, B4, q4,
     A11, B11, q11, A21, B21, q21, A31, B31, q31, A41, B41, q41,
     A51, B51, q51, A52, B52, q52, A53, B53, q53,
     A61, B61, q61, A62, B62, q62, A63, B63, q63,
     Kmax, Ku, nHill, halfmax, Kt, Vhalf, Temp, ko, amp) = P

    # 细胞外浓度 / 物理常数（C 码硬编码，照抄）
    nao = 140.0
    cao = 1.8
    R = 8314.0
    T = 310.0
    F = 96485.0

    # 几何（注意 C 码用 3.14 不是 pi）
    L = 0.01
    rad = 0.0011
    vcell = 1000 * 3.14 * rad * rad * L
    Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
    Acap = 2 * Ageo
    vmyo = 0.68 * vcell
    vnsr = 0.0552 * vcell
    vjsr = 0.0048 * vcell
    vss = 0.02 * vcell

    Acap_Fvmyo = Acap / (F * vmyo)
    Acap_Fvss = Acap / (F * vss)
    Acap_2Fvmyo = Acap / (2.0 * F * vmyo)
    Acap_2Fvss = Acap / (2.0 * F * vss)
    vss_vmyo = vss / vmyo
    vnsr_vmyo = vnsr / vmyo
    vjsr_vss = vjsr / vss
    vjsr_vnsr = vjsr / vnsr

    # hERG Markov 温度因子：rate = A*exp(B*v)*q^((Temp-20)/10)（Temp=37 -> q^1.7）
    tfac = (Temp - 20.0) / 10.0
    A1t = A1 * math.exp(tfac * math.log(q1));   A2t = A2 * math.exp(tfac * math.log(q2))
    A3t = A3 * math.exp(tfac * math.log(q3));   A4t = A4 * math.exp(tfac * math.log(q4))
    A11t = A11 * math.exp(tfac * math.log(q11)); A21t = A21 * math.exp(tfac * math.log(q21))
    A31t = A31 * math.exp(tfac * math.log(q31)); A41t = A41 * math.exp(tfac * math.log(q41))
    A51t = A51 * math.exp(tfac * math.log(q51)); A61t = A61 * math.exp(tfac * math.log(q61))
    A52t = A52 * math.exp(tfac * math.log(q52)); A62t = A62 * math.exp(tfac * math.log(q62))
    A53t = A53 * math.exp(tfac * math.log(q53)); A63t = A63 * math.exp(tfac * math.log(q63))

    # 电导（celltype=0 分支已解析，分支照抄）
    GNa = 75.0 / GNafc
    GNaL = 0.0075 / GNaLfc
    if celltype == 1:
        GNaL = GNaL * 0.6
    Gto = 0.02 / Gtofc
    if celltype == 1:
        Gto = Gto * 4.0
    elif celltype == 2:
        Gto = Gto * 4.0
    PCa = 0.0001 / PCafc
    if celltype == 1:
        PCa = PCa * 1.2
    elif celltype == 2:
        PCa = PCa * 2.5
    PCap = 1.1 * PCa
    PCaNa = 0.00125 * PCa
    PCaK = 3.574e-4 * PCa
    PCaNap = 0.00125 * PCap
    PCaKp = 3.574e-4 * PCap
    GKr = 0.046 / GKrfc
    if celltype == 1:
        GKr = GKr * 1.3
    elif celltype == 2:
        GKr = GKr * 0.8
    GKs = 0.0034 / GKsfc
    if celltype == 1:
        GKs = GKs * 1.4
    GK1 = 0.1908 / GK1fc
    if celltype == 1:
        GK1 = GK1 * 1.2
    elif celltype == 2:
        GK1 = GK1 * 1.3
    Gncx = 0.0008
    if celltype == 1:
        Gncx = Gncx * 1.1
    elif celltype == 2:
        Gncx = Gncx * 1.4
    Pnak = 30.0
    if celltype == 1:
        Pnak = Pnak * 0.9
    elif celltype == 2:
        Pnak = Pnak * 0.7
    GKb = 0.003
    if celltype == 1:
        GKb = GKb * 0.6
    GpCa = 0.0005
    PNab = 3.75e-10
    PCab = 2.5e-8

    epi_flag = 1.0 if celltype == 1 else 0.0      # delta_epi 只在 celltype==1 偏离 1
    caMKt_jrel = 1.7 if celltype == 2 else 1.0    # Jrel 因子
    jup_factor = 1.3 if celltype == 1 else 1.0
    cmdnmax = 0.05 * (1.3 if celltype == 1 else 1.0)

    sqrt_ko_54 = math.sqrt(ko / 5.4)
    sqrt_ko = math.sqrt(ko)
    RT_F = R * T / F
    F_RT = F / (R * T)
    F2_RT = F * F_RT

    def rhs(t, y):
        exp = math.exp
        if type(y) is not list:
            y = y.tolist()
        (v, nai, nass, ki, kss, cai, cass, cansr, cajsr,
         m, hf, hs, j, hsp, jp, mL, hL, hLp,
         a, iF, iS, ap, iFp, iSp,
         d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp,
         xs1, xs2, xk1, Jrelnp, Jrelp, CaMKt,
         IC1, IC2, C1, C2, O, IO, IObound, Obound, Cbound, D) = y[:49]

        dy = [0.0] * 50

        # ---- CaMK ----
        KmCaMK = 0.15
        aCaMK = 0.05
        bCaMK = 0.00068
        CaMKo = 0.05
        KmCaM = 0.0015
        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
        CaMKa = CaMKb + CaMKt
        dy[38] = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt

        # ---- 反转电位 ----
        ENa = RT_F * math.log(nao / nai)
        EK = RT_F * math.log(ko / ki)
        PKNa = 0.01833
        EKs = RT_F * math.log((ko + PKNa * nao) / (ki + PKNa * nai))

        vffrt = v * F2_RT
        vfrt = v * F_RT

        # ---- INa ----
        mss = 1.0 / (1.0 + exp((-(v + 39.57)) / 9.871))
        tm = 1.0 / (6.765 * exp((v + 11.64) / 34.77) + 8.552 * exp(-(v + 77.42) / 5.955))
        dy[9] = (mss - m) / tm
        hss = 1.0 / (1.0 + exp((v + 82.90) / 6.086))
        thf = 1.0 / (1.432e-5 * exp(-(v + 1.196) / 6.285) + 6.149 * exp((v + 0.5096) / 20.27))
        ths = 1.0 / (0.009794 * exp(-(v + 17.95) / 28.05) + 0.3343 * exp((v + 5.730) / 56.66))
        Ahf = 0.99
        Ahs = 1.0 - Ahf
        dy[10] = (hss - hf) / thf
        dy[11] = (hss - hs) / ths
        h = Ahf * hf + Ahs * hs
        jss = hss
        tj = 2.038 + 1.0 / (0.02136 * exp(-(v + 100.6) / 8.281) + 0.3052 * exp((v + 0.9941) / 38.45))
        dy[12] = (jss - j) / tj
        hssp = 1.0 / (1.0 + exp((v + 89.1) / 6.086))
        thsp = 3.0 * ths
        dy[13] = (hssp - hsp) / thsp
        hp = Ahf * hf + Ahs * hsp
        tjp = 1.46 * tj
        dy[14] = (jss - jp) / tjp
        fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
        INa = GNa * (v - ENa) * m ** 3.0 * ((1.0 - fINap) * h * j + fINap * hp * jp)

        # ---- INaL ----
        mLss = 1.0 / (1.0 + exp((-(v + 42.85)) / 5.264))
        tmL = tm
        dy[15] = (mLss - mL) / tmL
        hLss = 1.0 / (1.0 + exp((v + 87.61) / 7.488))
        thL = 200.0
        dy[16] = (hLss - hL) / thL
        hLssp = 1.0 / (1.0 + exp((v + 93.81) / 7.488))
        thLp = 3.0 * thL
        dy[17] = (hLssp - hLp) / thLp
        fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        INaL = GNaL * (v - ENa) * mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)

        # ---- Ito ----
        ass = 1.0 / (1.0 + exp((-(v - 14.34)) / 14.82))
        ta = 1.0515 / (1.0 / (1.2089 * (1.0 + exp(-(v - 18.4099) / 29.3814)))
                       + 3.5 / (1.0 + exp((v + 100.0) / 29.3814)))
        dy[18] = (ass - a) / ta
        iss = 1.0 / (1.0 + exp((v + 43.94) / 5.711))
        delta_epi = 1.0 - epi_flag * (0.95 / (1.0 + exp((v + 70.0) / 5.0)))
        tiF = 4.562 + 1.0 / (0.3933 * exp((-(v + 100.0)) / 100.0) + 0.08004 * exp((v + 50.0) / 16.59))
        tiS = 23.62 + 1.0 / (0.001416 * exp((-(v + 96.52)) / 59.05) + 1.780e-8 * exp((v + 114.1) / 8.079))
        tiF = tiF * delta_epi
        tiS = tiS * delta_epi
        AiF = 1.0 / (1.0 + exp((v - 213.6) / 151.2))
        AiS = 1.0 - AiF
        dy[19] = (iss - iF) / tiF
        dy[20] = (iss - iS) / tiS
        i_ = AiF * iF + AiS * iS
        assp = 1.0 / (1.0 + exp((-(v - 24.34)) / 14.82))
        dy[21] = (assp - ap) / ta
        dti_develop = 1.354 + 1.0e-4 / (exp((v - 167.4) / 15.89) + exp(-(v - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + exp((v + 70.0) / 20.0))
        tiFp = dti_develop * dti_recover * tiF
        tiSp = dti_develop * dti_recover * tiS
        dy[22] = (iss - iFp) / tiFp
        dy[23] = (iss - iSp) / tiSp
        myip = AiF * iFp + AiS * iSp
        fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
        Ito = Gto * (v - EK) * ((1.0 - fItop) * a * i_ + fItop * ap * myip)

        # ---- ICaL / ICaNa / ICaK ----
        dss = 1.0 / (1.0 + exp((-(v + 3.940)) / 4.230))
        td = 0.6 + 1.0 / (exp(-0.05 * (v + 6.0)) + exp(0.09 * (v + 14.0)))
        dy[24] = (dss - d) / td
        fss = 1.0 / (1.0 + exp((v + 19.58) / 3.696))
        tff = 7.0 + 1.0 / (0.0045 * exp(-(v + 20.0) / 10.0) + 0.0045 * exp((v + 20.0) / 10.0))
        tfs = 1000.0 + 1.0 / (0.000035 * exp(-(v + 5.0) / 4.0) + 0.000035 * exp((v + 5.0) / 6.0))
        Aff = 0.6
        Afs = 1.0 - Aff
        dy[25] = (fss - ff) / tff
        dy[26] = (fss - fs) / tfs
        f = Aff * ff + Afs * fs
        fcass = fss
        tfcaf = 7.0 + 1.0 / (0.04 * exp(-(v - 4.0) / 7.0) + 0.04 * exp((v - 4.0) / 7.0))
        tfcas = 100.0 + 1.0 / (0.00012 * exp(-v / 3.0) + 0.00012 * exp(v / 7.0))
        Afcaf = 0.3 + 0.6 / (1.0 + exp((v - 10.0) / 10.0))
        Afcas = 1.0 - Afcaf
        dy[27] = (fcass - fcaf) / tfcaf
        dy[28] = (fcass - fcas) / tfcas
        fca = Afcaf * fcaf + Afcas * fcas
        tjca = 75.0
        dy[29] = (fcass - jca) / tjca
        tffp = 2.5 * tff
        dy[31] = (fss - ffp) / tffp
        fp = Aff * ffp + Afs * fs
        tfcafp = 2.5 * tfcaf
        dy[32] = (fcass - fcafp) / tfcafp
        fcap = Afcaf * fcafp + Afcas * fcas
        Kmn = 0.002
        k2n = 1000.0
        km2n = jca * 1.0
        anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass) ** 4.0)
        dy[30] = anca * k2n - nca * km2n
        e2v = exp(2.0 * vfrt)
        den2 = e2v - 1.0
        if abs(den2) < 1e-12:
            PhiCaL = 2.0 * F * (cass - 0.341 * cao)          # v->0 极限（可去奇点保险）
        else:
            PhiCaL = 4.0 * vffrt * (cass * e2v - 0.341 * cao) / den2
        e1v = exp(vfrt)
        den1 = e1v - 1.0
        if abs(den1) < 1e-12:
            PhiCaNa = F * (0.75 * nass - 0.75 * nao)
            PhiCaK = F * (0.75 * kss - 0.75 * ko)
        else:
            PhiCaNa = vffrt * (0.75 * nass * e1v - 0.75 * nao) / den1
            PhiCaK = vffrt * (0.75 * kss * e1v - 0.75 * ko) / den1
        zca = 2.0
        fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        ICaL = ((1.0 - fICaLp) * PCa * PhiCaL * d * (f * (1.0 - nca) + jca * fca * nca)
                + fICaLp * PCap * PhiCaL * d * (fp * (1.0 - nca) + jca * fcap * nca))
        ICaNa = ((1.0 - fICaLp) * PCaNa * PhiCaNa * d * (f * (1.0 - nca) + jca * fca * nca)
                 + fICaLp * PCaNap * PhiCaNa * d * (fp * (1.0 - nca) + jca * fcap * nca))
        ICaK = ((1.0 - fICaLp) * PCaK * PhiCaK * d * (f * (1.0 - nca) + jca * fca * nca)
                + fICaLp * PCaKp * PhiCaK * d * (fp * (1.0 - nca) + jca * fcap * nca))

        # ---- IKr（hERG 六态 Markov + 药物结合态；control 下 Kmax=Ku=0）----
        r1f = A1t * exp(B1 * v);  r1b = A2t * exp(B2 * v)
        r2f = A3t * exp(B3 * v);  r2b = A4t * exp(B4 * v)
        r3f = A11t * exp(B11 * v); r3b = A21t * exp(B21 * v)
        r4f = A31t * exp(B31 * v); r4b = A41t * exp(B41 * v)
        r5f = A51t * exp(B51 * v); r5b = A61t * exp(B61 * v)
        r6f = A52t * exp(B52 * v); r6b = A62t * exp(B62 * v)
        r7f = A53t * exp(B53 * v); r7b = A63t * exp(B63 * v)
        j1 = r1f * C1 - r1b * C2
        j2 = r2f * IC2 - r2b * IO
        j3 = r3f * IC1 - r3b * IC2
        j4 = r4f * C2 - r4b * O
        j5 = r5f * C1 - r5b * IC1
        j6 = r6f * C2 - r6b * IC2
        j7 = r7f * O - r7b * IO
        Dn = D ** nHill if D > 0.0 else 0.0     # C 码 log(0)=-inf 语义：D=0 时结合通量为 0
        kon_v = Kmax * Ku * Dn / (Dn + halfmax)
        ktr = Kt / (1.0 + exp(-(v - Vhalf) / 6.789))
        dy[39] = -j3 + j5
        dy[40] = j3 - j2 + j6
        dy[41] = -j1 - j5
        dy[42] = j1 - j4 - j6
        dy[43] = j4 - j7 - (kon_v * O - Ku * Obound)
        dy[44] = j2 + j7 - (kon_v * IO - Ku * (r7f / r7b) * IObound)
        dy[45] = (kon_v * IO - Ku * (r7f / r7b) * IObound) + (ktr * Cbound - Kt * IObound)
        dy[46] = (kon_v * O - Ku * Obound) + (ktr * Cbound - Kt * Obound)
        dy[47] = -(ktr * Cbound - Kt * Obound) - (ktr * Cbound - Kt * IObound)
        dy[48] = 0.0
        IKr = GKr * sqrt_ko_54 * O * (v - EK)

        # ---- IKs ----
        xs1ss = 1.0 / (1.0 + exp((-(v + 11.60)) / 8.932))
        txs1 = 817.3 + 1.0 / (2.326e-4 * exp((v + 48.28) / 17.80) + 0.001292 * exp((-(v + 210.0)) / 230.0))
        dy[33] = (xs1ss - xs1) / txs1
        xs2ss = xs1ss
        txs2 = 1.0 / (0.01 * exp((v - 50.0) / 20.0) + 0.0193 * exp((-(v + 66.54)) / 31.0))
        dy[34] = (xs2ss - xs2) / txs2
        KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)
        IKs = GKs * KsCa * xs1 * xs2 * (v - EKs)

        # ---- IK1 ----
        xk1ss = 1.0 / (1.0 + exp(-(v + 2.5538 * ko + 144.59) / (1.5692 * ko + 3.8115)))
        txk1 = 122.2 / (exp((-(v + 127.2)) / 20.36) + exp((v + 236.8) / 69.33))
        dy[35] = (xk1ss - xk1) / txk1
        rk1 = 1.0 / (1.0 + exp((v + 105.8 - 2.6 * ko) / 9.493))
        IK1 = GK1 * sqrt_ko * rk1 * xk1 * (v - EK)

        # ---- INaCa_i / INaCa_ss（两块同构，钠钙浓度不同）----
        kna1 = 15.0; kna2 = 5.0; kna3 = 88.12; kasymm = 12.5
        wna = 6.0e4; wca = 6.0e4; wnaca = 5.0e3
        kcaon = 1.5e6; kcaoff = 5.0e3
        qna = 0.5224; qca = 0.1670
        hca = exp(qca * vfrt)
        hna = exp(qna * vfrt)
        KmCaAct = 150.0e-6

        # i 池
        h1 = 1.0 + nai / kna3 * (1.0 + hna)
        h2 = (nai * hna) / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + nai / kna1 * (1.0 + nai / kna2)
        h5 = nai * nai / (h4 * kna1 * kna2)
        h6 = 1.0 / h4
        h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
        h8 = nao / (kna3 * hna * h7)
        h9 = 1.0 / h7
        h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
        h11 = nao * nao / (h10 * kna1 * kna2)
        h12 = 1.0 / h10
        k1 = h12 * cao * kcaon
        k2 = kcaoff
        k3p = h9 * wca
        k3pp = h8 * wnaca
        k3 = k3p + k3pp
        k4p = h3 * wca / hca
        k4pp = h2 * wnaca
        k4 = k4p + k4pp
        k5 = kcaoff
        k6 = h6 * cai * kcaon
        k7 = h5 * h2 * wna
        k8 = h8 * h11 * wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        sE = x1 + x2 + x3 + x4
        E1 = x1 / sE; E2 = x2 / sE; E3 = x3 / sE; E4 = x4 / sE
        allo = 1.0 / (1.0 + (KmCaAct / cai) ** 2.0)
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa = E2 * k2 - E1 * k1
        INaCa_i = 0.8 * Gncx * allo * (1.0 * JncxNa + zca * JncxCa)

        # ss 池
        h1 = 1.0 + nass / kna3 * (1.0 + hna)
        h2 = (nass * hna) / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + nass / kna1 * (1.0 + nass / kna2)
        h5 = nass * nass / (h4 * kna1 * kna2)
        h6 = 1.0 / h4
        h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
        h8 = nao / (kna3 * hna * h7)
        h9 = 1.0 / h7
        h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
        h11 = nao * nao / (h10 * kna1 * kna2)
        h12 = 1.0 / h10
        k1 = h12 * cao * kcaon
        k3p = h9 * wca
        k3pp = h8 * wnaca
        k3 = k3p + k3pp
        k4p = h3 * wca / hca
        k4pp = h2 * wnaca
        k4 = k4p + k4pp
        k6 = h6 * cass * kcaon
        k7 = h5 * h2 * wna
        k8 = h8 * h11 * wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        sE = x1 + x2 + x3 + x4
        E1 = x1 / sE; E2 = x2 / sE; E3 = x3 / sE; E4 = x4 / sE
        allo = 1.0 / (1.0 + (KmCaAct / cass) ** 2.0)
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa = E2 * k2 - E1 * k1
        INaCa_ss = 0.2 * Gncx * allo * (1.0 * JncxNa + zca * JncxCa)

        # ---- INaK ----
        k1p = 949.5; k1m = 182.4; k2p = 687.2; k2m = 39.4
        k3p_nk = 1899.0; k3m_nk = 79300.0; k4p_nk = 639.0; k4m_nk = 40.0
        Knai0 = 9.073; Knao0 = 27.78; delta = -0.1550
        Knai = Knai0 * exp(delta * vfrt / 3.0)
        Knao = Knao0 * exp((1.0 - delta) * vfrt / 3.0)
        Kki = 0.5; Kko = 0.3582
        MgADP = 0.05; MgATP = 9.8; Kmgatp = 1.698e-7
        H = 1.0e-7; eP = 4.2; Khp = 1.698e-7; Knap = 224.0; Kxkur = 292.0
        Pn = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)
        a1 = (k1p * (nai / Knai) ** 3.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)
        b1 = k1m * MgADP
        a2 = k2p
        b2 = (k2m * (nao / Knao) ** 3.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
        a3 = (k3p_nk * (ko / Kko) ** 2.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
        b3 = (k3m_nk * Pn * H) / (1.0 + MgATP / Kmgatp)
        a4 = (k4p_nk * MgATP / Kmgatp) / (1.0 + MgATP / Kmgatp)
        b4 = (k4m_nk * (ki / Kki) ** 2.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)
        x1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
        x2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
        x3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
        x4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1
        sE = x1 + x2 + x3 + x4
        E1 = x1 / sE; E2 = x2 / sE; E3 = x3 / sE; E4 = x4 / sE
        JnakNa = 3.0 * (E1 * a3 - E2 * b3)
        JnakK = 2.0 * (E4 * b1 - E3 * a1)
        INaK = Pnak * (1.0 * JnakNa + 1.0 * JnakK)

        # ---- 背景 / 泵 ----
        xkb = 1.0 / (1.0 + exp(-(v - 14.48) / 18.34))
        IKb = GKb * xkb * (v - EK)
        if abs(den1) < 1e-12:
            INab = PNab * F * (nai - nao)
        else:
            INab = PNab * vffrt * (nai * e1v - nao) / den1
        if abs(den2) < 1e-12:
            ICab = PCab * 2.0 * F * (cai - 0.341 * cao)
        else:
            ICab = PCab * 4.0 * vffrt * (cai * e2v - 0.341 * cao) / den2
        IpCa = GpCa * cai / (0.0005 + cai)

        # ---- 刺激（拍内 t <= 0.5 ms）----
        Istim = amp if t <= DUR else 0.0

        # ---- 膜电位 ----
        dy[0] = -(INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1
                  + INaCa_i + INaCa_ss + INaK + INab + IKb + IpCa + ICab + Istim)

        # ---- 扩散通量 ----
        JdiffNa = (nass - nai) / 2.0
        JdiffK = (kss - ki) / 2.0
        Jdiff = (cass - cai) / 0.2

        # ---- RyR 释放 ----
        bt = 4.75
        a_rel = 0.5 * bt
        Jrel_inf = a_rel * (-ICaL) / (1.0 + (1.5 / cajsr) ** 8.0) * caMKt_jrel
        tau_rel = bt / (1.0 + 0.0123 / cajsr)
        if tau_rel < 0.001:
            tau_rel = 0.001
        dy[36] = (Jrel_inf - Jrelnp) / tau_rel
        btp = 1.25 * bt
        a_relp = 0.5 * btp
        Jrel_infp = a_relp * (-ICaL) / (1.0 + (1.5 / cajsr) ** 8.0) * caMKt_jrel
        tau_relp = btp / (1.0 + 0.0123 / cajsr)
        if tau_relp < 0.001:
            tau_relp = 0.001
        dy[37] = (Jrel_infp - Jrelp) / tau_relp
        fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
        Jrel = (1.0 - fJrelp) * Jrelnp + fJrelp * Jrelp

        # ---- SERCA 摄取 ----
        Jupnp = 0.004375 * cai / (cai + 0.00092) * jup_factor
        Jupp = 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017) * jup_factor
        fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
        Jleak = 0.0039375 * cansr / 15.0
        Jup = (1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak

        Jtr = (cansr - cajsr) / 100.0

        # ---- 缓冲与浓度 ----
        kmcmdn = 0.00238
        trpnmax = 0.07
        kmtrpn = 0.0005
        BSRmax = 0.047
        KmBSR = 0.00087
        BSLmax = 1.124
        KmBSL = 0.0087
        csqnmax = 10.0
        kmcsqn = 0.8

        dy[1] = -(INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab) * Acap_Fvmyo + JdiffNa * vss_vmyo
        dy[2] = -(ICaNa + 3.0 * INaCa_ss) * Acap_Fvss - JdiffNa
        dy[3] = -(Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK) * Acap_Fvmyo + JdiffK * vss_vmyo
        dy[4] = -(ICaK) * Acap_Fvss - JdiffK

        Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + cai) ** 2.0
                      + trpnmax * kmtrpn / (kmtrpn + cai) ** 2.0)
        dy[5] = Bcai * (-(IpCa + ICab - 2.0 * INaCa_i) * Acap_2Fvmyo
                        - Jup * vnsr_vmyo + Jdiff * vss_vmyo)
        Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + cass) ** 2.0
                       + BSLmax * KmBSL / (KmBSL + cass) ** 2.0)
        dy[6] = Bcass * (-(ICaL - 2.0 * INaCa_ss) * Acap_2Fvss + Jrel * vjsr_vss - Jdiff)
        dy[7] = Jup - Jtr * vjsr_vnsr
        Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr) ** 2.0)
        dy[8] = Bcajsr * (Jtr - Jrel)

        # ---- qNet 积分器（官方六电流，不含 INa）----
        dy[49] = INaL + ICaL + Ito + IKr + IKs + IK1

        return dy, (INa, INaL, Ito, ICaL, IKr, IKs, IK1)

    return rhs


# ----------------------------------------------------------------------------
# APD90（官方 metric_funs.R find_rep 口径：xrest=拍内 v 最小，t_rep − t_dVdtmax）
# ----------------------------------------------------------------------------
def apd90(t, v):
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    apa = vmax - vmin
    if apa < 20.0:
        return float("nan"), apa, float("nan"), float("nan")
    dt = t[1] - t[0]
    dvd = np.diff(v) / dt
    iup = int(np.argmax(dvd))
    t_up = float(t[iup])
    v90 = vmin + 0.1 * apa
    idx = np.where((v < v90) & (t > t_up))[0]
    if len(idx) == 0:
        return float("nan"), apa, t_up, float("nan")
    j0 = int(idx[0])
    frac = (v[j0 - 1] - v90) / (v[j0 - 1] - v[j0])
    t_rep = float(t[j0 - 1] + frac * dt)
    return t_rep - t_up, apa, t_up, t_rep


# ----------------------------------------------------------------------------
# Euler 交叉对拍臂（前两拍，dt=0.005 ms，与 LSODA 同 rhs）
# ----------------------------------------------------------------------------
def euler_beats(rhs, y0, nbe, dt, t_rec):
    y = list(y0)
    nsteps = int(round(CL / dt))
    keep = int(round(DT_REC / dt))
    out = []
    for _ in range(nbe):
        vrec = np.empty(len(t_rec))
        for s in range(nsteps):
            t = s * dt
            dydt, _ = rhs(t, y)
            if s % keep == 0:
                vrec[s // keep] = y[0]
            for i in range(50):
                y[i] += dt * dydt[i]
        vrec[len(t_rec) - 1] = y[0]
        out.append(vrec)
    return out


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" α模型 Stage B · B0 地基复核（官方 newordherg_qNet.c Python 移植，control）", flush=True)
    print(f" 模式: {'冒烟 SMOKE（100 拍）' if SMOKE else '正式（' + str(NBEATS) + ' 拍）'}  CL={CL:.0f}ms  记录网格 {DT_REC}ms", flush=True)
    print(" 积分器: LSODA rtol=1e-7 atol=1e-9（v2 修订）；Euler dt=0.005ms 前两拍交叉对拍", flush=True)
    print("=" * 74, flush=True)

    P = load_named_values(PARS_FILE, PARS_NAMES)
    y0 = load_named_values(STATES_FILE, STATE_NAMES) + [0.0]     # 第 50 态 qNet 积分器
    assert abs(P[0] - 0.0) < 1e-12 and abs(P[1] - 1.0) < 1e-12, "非 control 参数档！"
    print(f"[锚] 参数 {len(P)} 项（celltype=0 control, Temp={P[56]:.0f}, ko={P[57]}, amp={P[58]}），"
          f"初值 {len(y0) - 1} 态 + qNet（CL2000 稳态起跳）", flush=True)

    rhs = build_model(P)
    from scipy.integrate import solve_ivp
    f = lambda t, y: rhs(t, y)[0]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)

    # ---------- Euler 交叉对拍（判线4 v3：拍1 双步长 + LSODA 参考自洽）----------
    print("\n[判线4·交叉对拍] Euler 拍1 dt=0.005/0.00125ms + LSODA(1e-9) 参考 ...", flush=True)
    t0 = time.time()
    eu_c = euler_beats(rhs, y0, 1, 0.005, t_rec)[0]
    print(f"  Euler dt=0.005 拍1完成，累计 {time.time() - t0:.1f}s", flush=True)
    eu_f = euler_beats(rhs, y0, 1, 0.00125, t_rec)[0]
    print(f"  Euler dt=0.00125 拍1完成，累计 {time.time() - t0:.1f}s", flush=True)
    sol_ref = solve_ivp(f, (0.0, CL), np.array(y0, dtype=float), method="LSODA",
                        t_eval=t_rec, rtol=1e-9, atol=1e-11)
    v_ref = sol_ref.y[0]
    print(f"  LSODA(1e-9) 参考拍完成，累计 {time.time() - t0:.1f}s", flush=True)

    # ---------- LSODA 主循环 ----------
    y = np.array(y0, dtype=float)
    apd = np.full(NBEATS, np.nan)
    qnet = np.full(NBEATS, np.nan)
    vmin_a = np.full(NBEATS, np.nan)
    vmax_a = np.full(NBEATS, np.nan)
    lsoda_first2 = []
    last2 = []          # 末两拍 (t, v, currents[7])
    nfail = 0
    print("\n[起搏]", flush=True)
    for b in range(NBEATS):
        q0 = y[49]
        sol = solve_ivp(f, (0.0, CL), y, method="LSODA",
                        t_eval=t_rec, rtol=1e-7, atol=1e-9)
        if not sol.success:
            nfail += 1
            print(f"  拍 {b + 1}: LSODA 失败 {sol.message}", flush=True)
            break
        v = sol.y[0]
        y = sol.y[:, -1].copy()
        a90, apa, t_up, t_rep = apd90(t_rec, v)
        apd[b] = a90
        qnet[b] = (y[49] - q0) * 1e-3
        vmin_a[b] = float(np.min(v))
        vmax_a[b] = float(np.max(v))
        if b < 2:
            lsoda_first2.append(v.copy())
        if b >= NBEATS - 2:
            cur = np.empty((7, len(t_rec)))
            for k in range(len(t_rec)):
                _, c7 = rhs(float(t_rec[k]), sol.y[:, k].tolist())
                cur[:, k] = c7
            last2.append((t_rec.copy(), v.copy(), cur, t_up, t_rep))
        if (b + 1) % 20 == 0 or b == 0:
            el = time.time() - t_start
            eta = el / (b + 1) * (NBEATS - b - 1)
            print(f"  拍 {b + 1}/{NBEATS}  APD90={a90:7.2f}ms  qNet={qnet[b]:.4f}µC/µF  "
                  f"v=[{vmin_a[b]:.1f},{vmax_a[b]:.1f}]  用时{el:.0f}s ETA{eta:.0f}s", flush=True)

    ndone = int(np.sum(~np.isnan(apd)))
    # ---------- 判线4（v3 收敛签名）----------
    xcheck = {}
    if len(lsoda_first2) >= 1:
        v_ls = lsoda_first2[0]
        apa_ref = float(np.max(v_ls) - np.min(v_ls))
        d_c = np.abs(eu_c - v_ls)
        d_f = np.abs(eu_f - v_ls)
        e_c = float(np.max(d_c))
        e_f = float(np.max(d_f))
        t_at = float(t_rec[int(np.argmax(d_f))])
        ratio = e_c / e_f if e_f > 0 else float("inf")
        ref_diff = float(np.max(np.abs(v_ref - v_ls)))
        a_ls, _, _, _ = apd90(t_rec, v_ls)
        a_eu, _, _, _ = apd90(t_rec, eu_f)
        dAPD = abs(a_ls - a_eu) if not (math.isnan(a_ls) or math.isnan(a_eu)) else float("nan")
        ok = bool(2.0 <= ratio <= 6.0 and e_f < 5.0 and t_at <= 10.0
                  and dAPD < 1.0 and ref_diff < 0.5)
        xcheck = {"max_abs_dv_mV": e_c, "max_abs_dv_fine_mV": e_f, "ratio": ratio,
                  "argmax_t_ms": t_at, "dAPD90_ms": dAPD, "lsoda_ref_diff_mV": ref_diff,
                  "APA_ref_mV": apa_ref, "rel": e_f / apa_ref if apa_ref > 0 else float("nan"),
                  "pass": ok}
    print(f"\n[判线4] 收敛签名: e005={xcheck.get('max_abs_dv_mV', float('nan')):.3f}mV "
          f"e00125={xcheck.get('max_abs_dv_fine_mV', float('nan')):.3f}mV "
          f"比={xcheck.get('ratio', float('nan')):.2f}∈[2,6] "
          f"峰位={xcheck.get('argmax_t_ms', float('nan')):.1f}ms(≤10) "
          f"ΔAPD90={xcheck.get('dAPD90_ms', float('nan')):.3f}ms(<1) "
          f"LSODA参考差={xcheck.get('lsoda_ref_diff_mV', float('nan')):.4f}mV(<0.5) "
          f"-> {'过' if xcheck.get('pass') else '不过'}", flush=True)

    # ---------- 判线1 稳态化 ----------
    last10 = apd[max(0, ndone - 10):ndone]
    steady_range = float(np.nanmax(last10) - np.nanmin(last10)) if ndone >= 10 else float("nan")
    steady_pass = bool(steady_range < 0.5)
    print(f"[判线1] 末10拍 APD90 极差 = {steady_range:.3f} ms (<0.5) -> {'过' if steady_pass else '不过'}"
          + ("" if steady_pass else "（达不到→登记并延长 NBEATS，判线不动）"), flush=True)

    # ---------- 判线2 形态 ----------
    ib = ndone - 1
    APD_last = float(apd[ib])
    qNet_last = float(qnet[ib])
    RMP_last = float(vmin_a[ib])
    APA_last = float(vmax_a[ib] - vmin_a[ib])
    crit2 = {
        "APD90_ms": {"value": APD_last, "band": [250.0, 330.0], "pass": bool(250.0 <= APD_last <= 330.0)},
        "qNet_uC_uF": {"value": qNet_last, "band": [0.05, 0.12], "pass": bool(0.05 <= qNet_last <= 0.12)},
        "RMP_mV": {"value": RMP_last, "band": [-95.0, -85.0], "pass": bool(-95.0 <= RMP_last <= -85.0)},
        "APA_mV": {"value": APA_last, "band": [90.0, 130.0], "pass": bool(90.0 <= APA_last <= 130.0)},
    }
    crit2_pass = all(c["pass"] for c in crit2.values())
    print("[判线2] 末拍形态：" + "  ".join(
        f"{k}={c['value']:.3f}∈[{c['band'][0]},{c['band'][1]}]{'✓' if c['pass'] else '×'}"
        for k, c in crit2.items()) + f" -> {'过' if crit2_pass else '不过'}", flush=True)

    # ---------- 判线3 qNet 双口径（末拍 AP 窗内积分）----------
    qnet_ap = float("nan")
    if last2:
        tL, vL, curL, t_up, t_rep = last2[-1]
        sum6 = curL[1] + curL[2] + curL[3] + curL[4] + curL[5] + curL[6]  # INaL+Ito+ICaL+IKr+IKs+IK1
        if not (math.isnan(t_up) or math.isnan(t_rep)):
            win = (tL >= t_up) & (tL <= t_rep)
            qnet_ap = float(np.trapezoid(sum6[win], tL[win]) * 1e-3)
    print(f"[判线3] qNet 整周期={qNet_last:.4f} µC/µF（状态49差分）  AP窗内={qnet_ap:.4f} µC/µF"
          f"（六电流积分，登记双口径）", flush=True)

    overall = bool(steady_pass and crit2_pass and xcheck.get("pass", False))
    print("\n" + "=" * 74, flush=True)
    print(f" B0 总判词：{'过线 —— Python 引擎与官方口径等价，Stage B 可进 B1' if overall else '不过线 —— Stage B 不开工，查移植'}", flush=True)
    if not SMOKE and steady_pass is False:
        print(" 提示：稳态化未达 → 以环境变量 NBEATS=2000 重跑（判线不动，照实登记）。", flush=True)
    print("=" * 74, flush=True)

    # ---------- 落盘 ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(BASE, f"2026-09-14_α模型_StageB_B0_ORd移植地基复核{tag}.json")
    out = {
        "meta": {"smoke": SMOKE, "nbeats_done": ndone, "CL_ms": CL, "dt_rec_ms": DT_REC,
                 "integrator": "LSODA rtol=1e-7 atol=1e-9", "euler_check_dt_ms": 0.005,
                 "pars_file": PARS_FILE, "states_file": STATES_FILE,
                 "runtime_s": time.time() - t_start, "lsoda_fail": nfail},
        "xcheck_euler_vs_lsoda": xcheck,
        "per_beat": {"apd90_ms": [None if math.isnan(x) else float(x) for x in apd[:ndone]],
                     "qnet_uC_uF": [None if math.isnan(x) else float(x) for x in qnet[:ndone]],
                     "vmin_mV": [float(x) for x in vmin_a[:ndone]],
                     "vmax_mV": [float(x) for x in vmax_a[:ndone]]},
        "final_beat": {"APD90_ms": APD_last, "qNet_cycle_uC_uF": qNet_last,
                       "qNet_APwindow_uC_uF": qnet_ap, "RMP_mV": RMP_last, "APA_mV": APA_last},
        "judgments": {"J1_steady": {"last10_range_ms": steady_range, "pass": steady_pass},
                      "J2_morphology": crit2, "J2_pass": crit2_pass,
                      "J4_xcheck_pass": xcheck.get("pass", False),
                      "B0_overall_pass": overall},
    }
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n 结果落盘: {fjson}", flush=True)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(17, 15))
    fig.suptitle(f"B0 地基复核 · 官方 ORd Python 移植 · {'冒烟100拍' if SMOKE else str(ndone) + '拍'}"
                 f" · 总判 {'过' if overall else '不过'}", fontsize=14)

    ax = axes[0, 0]
    if last2:
        tL, vL, _, _, _ = last2[-1]
        ax.plot(tL, vL, "k-", lw=1.2)
    ax.set_title("末拍 AP 全曲线")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("v (mV)")

    ax = axes[0, 1]
    if last2:
        msk = tL <= 60.0
        ax.plot(tL[msk], vL[msk], "k-", lw=1.2)
    ax.set_title("末拍前 60ms（上冲+早期复极）")
    ax.set_xlabel("t (ms)")

    ax = axes[0, 2]
    ax.plot(np.arange(1, ndone + 1), apd[:ndone], ".-", ms=3, lw=0.6)
    ax.axhline(250, color="r", ls="--", lw=0.7); ax.axhline(330, color="r", ls="--", lw=0.7)
    ax.set_title(f"APD90 逐拍轨迹（末10拍极差 {steady_range:.2f}ms）")
    ax.set_xlabel("拍"); ax.set_ylabel("APD90 (ms)")

    ax = axes[1, 0]
    ax.plot(np.arange(1, ndone + 1), qnet[:ndone], ".-", ms=3, lw=0.6, color="tab:green")
    ax.axhline(0.05, color="r", ls="--", lw=0.7); ax.axhline(0.12, color="r", ls="--", lw=0.7)
    ax.set_title("qNet 逐拍轨迹（整周期口径）")
    ax.set_xlabel("拍"); ax.set_ylabel("µC/µF")

    names7 = ["INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1"]
    pos = [(1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]
    if last2:
        tL, vL, curL, _, _ = last2[-1]
        for (r, c), nm in zip(pos, names7):
            ax = axes[r, c]
            i7 = names7.index(nm)
            ax.plot(tL, curL[i7], lw=0.9)
            ax.set_title(f"{nm} 末拍")
            ax.set_xlabel("t (ms)"); ax.set_ylabel("pA/pF")
            if nm == "INa":
                ax.set_xlim(0, 30)

    ax = axes[3, 2]
    ax.axis("off")
    txt = (f"判线1 稳态化: 末10拍APD90极差 {steady_range:.3f}ms (<0.5) {'✓' if steady_pass else '×'}\n"
           f"判线2 形态: APD90 {APD_last:.1f}ms∈[250,330] {'✓' if crit2['APD90_ms']['pass'] else '×'}  "
           f"qNet {qNet_last:.3f}∈[0.05,0.12] {'✓' if crit2['qNet_uC_uF']['pass'] else '×'}\n"
           f"        RMP {RMP_last:.1f}mV∈[-95,-85] {'✓' if crit2['RMP_mV']['pass'] else '×'}  "
           f"APA {APA_last:.1f}mV∈[90,130] {'✓' if crit2['APA_mV']['pass'] else '×'}\n"
           f"判线3 qNet双口径: 整周期 {qNet_last:.4f} / AP窗内 {qnet_ap:.4f} µC/µF（登记）\n"
           f"判线4 Euler收敛签名: e005={xcheck.get('max_abs_dv_mV', float('nan')):.2f} "
           f"e00125={xcheck.get('max_abs_dv_fine_mV', float('nan')):.2f}mV "
           f"比{xcheck.get('ratio', float('nan')):.2f}∈[2,6] "
           f"ΔAPD90={xcheck.get('dAPD90_ms', float('nan')):.2f}ms "
           f"{'✓' if xcheck.get('pass') else '×'}\n"
           f"LSODA失败 {nfail} 次   总用时 {time.time() - t_start:.0f}s\n"
           f"【B0 总判 {'过线' if overall else '不过线'}】")
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10)

    fpng = os.path.join(BASE, f"2026-09-14_α模型_StageB_B0_ORd移植地基复核{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" 图落盘: {fpng}", flush=True)

    if SMOKE:
        print("\n[冒烟完] 正式跑指令（Spyder）：\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
