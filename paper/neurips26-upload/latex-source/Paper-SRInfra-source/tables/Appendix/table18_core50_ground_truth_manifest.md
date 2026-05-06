# Core-50 ground-truth formula manifest

| Core50 index | Dataset | Family | Subgroup | Target | Features | Ground-truth math expression |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | II.34.2_1_0 | llm-srbench | lsrtransform | v | mom, q, r | 2 * mom / (q * r) |
| 2 | first_principles_hubble | srbench2025 | firstprinciples | target | D | 73.3 * D |
| 3 | feynman-ii.27.18 | srsd | srsd-feynman_easy_dummy | y | x0, x1, x2 | 8.854e-12 * x0 ** 2 |
| 4 | Korns-4 | korns | Korns-4 | target | x1, x2, x3, x4, x5 | 0.13 * sin(x3) - 2.3 |
| 5 | Keijzer-11 | keijzer | Keijzer-11 | target | x1, x2 | x1 * x2 + sin((x1 - 1) * (x2 - 1)) |
| 6 | Keijzer-2 | keijzer | Keijzer-2 | target | x1 | 0.3 * x1 * sin(2 * pi * x1) |
| 7 | Korns-2 | korns | Korns-2 | target | x1, x2, x3, x4, x5 | 0.23 + 14.2 * div(x4 + x2, 3 * x5) |
| 8 | Nguyen-12 | nguyen | Nguyen-12 | target | x1, x2 | pow(x1, 4) - pow(x1, 3) + div(pow(x2, 2), 2) - x2 |
| 9 | Nguyen-6 | nguyen | Nguyen-6 | target | x1 | sin(x1) + sin(x1 + pow(x1, 2)) |
| 10 | Nguyen-9 | nguyen | Nguyen-9 | target | x1, x2 | sin(x1) + sin(pow(x2, 2)) |
| 11 | Vladislavleva-4 | vladislavleva | Vladislavleva-4 | target | x1, x2, x3, x4, x5 | div(10, 5 + (pow(x1 - 3, 2) + pow(x2 - 3, 2) + pow(x3 - 3, 2) + pow(x4 - 3, 2) + pow(x5 - 3, 2))) |
| 12 | feynman_I_27_6 | srbench1.0 | feynman | target | d1, d2, n | 1 / (1 / d1 + n / d2) |
| 13 | feynman-i.43.31 | srsd | srsd-feynman_medium | y | x0, x1 | 1.380649e-23 * x0 * x1 |
| 14 | feynman-ii.34.11 | srsd | srsd-feynman_easy_dummy | y | x0, x1, x2, x3, x4, x5, x6 | x0 * x2 * x4 / (2 * x5) |
| 15 | III.21.20_3_0 | llm-srbench | lsrtransform | m | j, rho_c_0, q, A_vec | -A_vec * q * rho_c_0 / j |
| 16 | feynman_II_15_4 | srbench1.0 | feynman | target | mom, B, theta | -mom * B * cos(theta) |
| 17 | strogatz_shearflow1 | srbench1.0 | strogatz | target | x, y | 1 / tan(y) * cos(x) |
| 18 | feynman-i.18.16 | srsd | srsd-feynman_easy_dummy | y | x0, x1, x2, x3, x4, x5 | x0 * x1 * x3 * sin(x4) |
| 19 | feynman-ii.34.2a | srsd | srsd-feynman_medium_dummy | y | x0, x1, x2, x3, x4 | x1 * x3 / (2 * pi * x4) |
| 20 | first_principles_leavitt | srbench2025 | firstprinciples | target | logP | -2.084 * logP + 15.65 |
| 21 | feynman-i.39.22 | srsd | srsd-feynman_hard_dummy | y | x0, x1, x2, x3, x4 | 1.380649e-23 * x1 * x3 / x4 |
| 22 | feynman-i.11.19 | srsd | srsd-feynman_medium_dummy | y | x0, x1, x2, x3, x4, x5, x6, x7 | x0 * x1 + x2 * x3 + x5 * x6 |
| 23 | I.11.19_4_0 | llm-srbench | lsrtransform | y2 | A, x1, x2, x3, y1, y3 | (-A + x1 * y1 - x3 * y3) / x2 |
| 24 | strogatz_predprey2 | srbench1.0 | strogatz | target | x, y | y * (x / (1 + x) - 0.075 * y) |
| 25 | strogatz_barmag2 | srbench1.0 | strogatz | target | x, y | 0.5 * sin(y - x) - sin(y) |
| 26 | strogatz_barmag1 | srbench1.0 | strogatz | target | x, y | 0.5 * sin(x - y) - sin(x) |
| 27 | feynman_I_12_11 | srbench1.0 | feynman | target | q, Ef, B, v, theta | q * (Ef + B * v * sin(theta)) |
| 28 | feynman-ii.4.23 | srsd | srsd-feynman_easy | y | x0, x1 | 28235825615.541 * x0 / (pi * x1) |
| 29 | strogatz_vdp1 | srbench1.0 | strogatz | target | x, y | 10 * (y - 1 / 3 * x ** 3 + 1 / 3 * x) |
| 30 | feynman-iii.7.38 | srsd | srsd-feynman_easy | y | x0, x1 | 6.03682463024449e+33 * pi * x0 * x1 |
| 31 | II.11.27_1_0 | llm-srbench | lsrtransform | alpha | Pol, n, epsilon, Ef | 3 * Pol / (n * (3 * Ef * epsilon + Pol)) |
| 32 | feynman-i.13.12 | srsd | srsd-feynman_medium | y | x0, x1, x2, x3 | 6.6743e-11 * x0 * x1 * (-1 / x3 + 1 / x2) |
| 33 | feynman-iii.10.19 | srsd | srsd-feynman_hard_dummy | y | x0, x1, x2, x3, x4, x5 | x2 * sqrt(x3 ** 2 + x4 ** 2 + x5 ** 2) |
| 34 | CRK11 | llm-srbench | chem_react | dA_dt | t, A | -0.8817392153705143 * A ** 2 + 0.8817392153705143 * sin(sqrt(A)) |
| 35 | feynman-i.32.5 | srsd | srsd-feynman_medium | y | x0, x1 | 6.9862982685735e-16 * x0 ** 2 * x1 ** 2 / pi |
| 36 | CRK10 | llm-srbench | chem_react | dA_dt | t, A | -0.17495597007573369 * A ** 2 + 0.17495597007573369 * sin(log(A + 1)) |
| 37 | CRK34 | llm-srbench | chem_react | dA_dt | t, A | -0.1689114325901851 * A ** 2 + 0.1689114325901851 * A ** 0.3333333333333333 |
| 38 | PO19 | llm-srbench | phys_osc | dv_dt | x, t, v | -2 * 0.1 * v - 1.0 ** 2 * x * exp(-abs(x)) |
| 39 | CRK0 | llm-srbench | chem_react | dA_dt | t, A | -0.18997742423620262 * A ** 2 + 0.18997742423620262 * A ** 2 / (0.7497988950401423 * A ** 4 + 1) |
| 40 | feynman-i.29.16 | srsd | srsd-feynman_hard | y | x0, x1, x2, x3 | sqrt(x0 ** 2 + 2 * x0 * x1 * cos(x2 - x3) + x1 ** 2) |
| 41 | PO41 | llm-srbench | phys_osc | dv_dt | x, t, v | -0.3254156367450257 * (1 - x ** 2) * v - 0.3333333333333333 ** 2 * x * exp(-abs(x)) |
| 42 | feynman_test_15 | srbench1.0 | feynman | target | c, v, omega, theta | sqrt(1 - v ** 2 / c ** 2) * omega / (1 + v / c * cos(theta)) |
| 43 | CRK33 | llm-srbench | chem_react | dA_dt | t, A | -0.11846873325639269 * sqrt(A) - 0.11846873325639269 * A ** 2 + 0.11846873325639269 * t * sin(log(A + 1)) |
| 44 | feynman-i.15.3t | srsd | srsd-feynman_hard_dummy | y | x0, x1, x2, x3, x4 | (x1 - 1.11265005605362e-17 * x3 * x4) / sqrt(1 - 1.11265005605362e-17 * x3 ** 2) |
| 45 | feynman-bonus.20 | srsd | srsd-feynman_hard_dummy | y | x0, x1, x2, x3, x4, x5 | 7.83707760458308e-29 * x2 ** 2 * (x2 / x3 - sin(x5) ** 2 + x3 / x2) / (pi * x3 ** 2) |
| 46 | II.6.15b_3_0 | llm-srbench | lsrtransform | r | Ef, epsilon, p_d, theta | 6 ** (1 / 3) * (p_d * sin(theta) * cos(theta) / (Ef * epsilon)) ** (1 / 3) / (2 * pi ** (1 / 3)) |
| 47 | BPG5 | llm-srbench | bio_pop_growth | dP_dt | t, P | 0.9198344660444582 * (1 - P / 84.0283518630852) * P + 0.9198344660444582 * P ** 2 / (7.531475575495239 * P + 1) |
| 48 | MatSci21 | llm-srbench | matsci | sigma | epsilon, T | 28.580443564163904 * epsilon ** 2 - 0.28570034656521326 * (T - 281.87025337570554) + epsilon ** 3 * 4.278428749432679 * (T - 281.87025337570554) |
| 49 | BPG3 | llm-srbench | bio_pop_growth | dP_dt | t, P | 0.8450821034088298 * (-1 + P / 5.115386454523557) * (1 - P / 34.43886242377677) * P + 0.8450821034088298 * (1 - exp(-0.09687266473516928 * P)) * P |
| 50 | BPG9 | llm-srbench | bio_pop_growth | dP_dt | t, P | 0.1699888096710614 * (-1 + P / 1.0472843670254641) * (1 - P / 10.210545156320043) * P + 0.1699888096710614 * (1 - P / 10.210545156320043) * P + 0.1699888096710614 * (1 - exp(-0.09709294493863778 * P)) * P |
