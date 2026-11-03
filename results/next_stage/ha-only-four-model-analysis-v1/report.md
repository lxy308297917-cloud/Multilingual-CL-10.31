# Hausa-only与顺序FFT：正确起点/终点及配对对照

| Task/subset | n | Base A0 | HA-only A1 | SW FFT B0 | SW→HA B1 | A gain | B gain | Final A−B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ha-belebele/all | 900 | 23.89 | 24.44 | 28.44 | 26.89 | +0.56 | -1.56 | -2.44 |
| ha-belebele/development_overlap180 | 180 | 29.44 | 28.89 | 31.67 | 29.44 | -0.56 | -2.22 | -0.56 |
| ha-belebele/not_in_current_curves720 | 720 | 22.50 | 23.33 | 27.64 | 26.25 | +0.83 | -1.39 | -2.92 |
| ha-gmmlu/all | 14042 | 26.25 | 27.52 | 29.72 | 27.65 | +1.27 | -2.07 | -0.13 |

全部区间、初始差、收益差和预定义00/01/10/11条件组见analysis.json。收益差含起点headroom，不单独当作负向迁移证明。阅读720仅与本轮曲线不重叠，不称历史未见。
