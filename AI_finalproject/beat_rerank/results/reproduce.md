## CLAP 文字-音樂相似度（9 個描述）

| | 對角線平均 | 非對角線平均 | 文字→音樂 R@1 | 音樂→文字 R@1 |
|---|---|---|---|---|
| 原報告 Fig. 1 | 0.269 | 0.245 | 0.00 | 0.33 |
| 重現（4 個樣本平均 ± 標準差） | 0.489 ± 0.007 | 0.340 ± 0.010 | 0.36 ± 0.09 | 0.39 ± 0.06 |

R@1 的隨機水準是 1/9 = 0.11。

## 音樂曲風分類（GTZAN 分類器）

| 描述 | 4 個樣本的預測 | 目標曲風機率（平均） |
|---|---|---|
| The audio is a Latin pop song. (text→music+motion) | pop, rock, disco, disco | pop 0.18 |
| The music is a mix of indie pop and indie rock. (text→music+motion) | rock, rock, metal, classical | rock/pop 0.20 |
| The music is pop genre. (text→music+motion) | rock, hiphop, rock, jazz | pop 0.04 |
| The music is disco genre. (text→music+motion) | disco, disco, disco, disco | disco 0.89 |
| The audio is a dreamy indie pop song with jangly guitars, airy vocals, and shimmering synth pads. (text→music+motion) | rock, disco, rock, classical | pop 0.02 |
| The audio is a gritty blues-rock track with overdriven guitar riffs, raw vocals, and a shuffling shuffle beat. (text→music+motion) | classical, metal, metal, classical | blues/rock 0.00 |
| The audio is a calm classical piano piece. (text→music+motion) | classical, classical, classical, classical | classical 0.97 |
| The audio is an energetic hip-hop beat with heavy bass and rap vocals. (text→music+motion) | reggae, hiphop, hiphop, hiphop | hiphop 0.71 |
| The audio is a fast heavy metal song with distorted guitars and pounding drums. (text→music+motion) | metal, metal, metal, metal | metal 0.96 |
| The audio is a gritty blues-rock track with overdriven guitar riffs, raw vocals, and a shuffling shuffle beat. (text+motion→music) | rock, metal, metal, classical | blues/rock 0.11 |
| The audio is a dreamy indie pop song with jangly guitars, airy vocals, and shimmering synth pads. (text+motion→music) | classical, rock, classical, classical | pop 0.01 |

## Beat Alignment Score（9 個 text + music → motion 結果）

| | BAS（onset，Bailando 定義） | ±5 幀命中率 | BAS（librosa beat） | onset BAS 減去配錯音樂的 onset BAS |
|---|---|---|---|---|
| 原報告 | 0.262 | | | |
| 重現，每個描述第 1 個樣本 | 0.207 | 0.298 | 0.216 | |
| 重現，全部 36 個樣本 | 0.222 ± 0.061 | 0.321 ± 0.092 | 0.231 ± 0.094 | -0.003 ± 0.034 |
| 真人舞者（AIST++，130 段） | 0.232 ± 0.054 | 0.341 ± 0.083 | 0.284 ± 0.129 | 0.008 ± 0.030 |

把音樂相對真人舞者平移 -0.75 到 +0.75 秒時的 BAS（同步的指標應該在 0 秒最高）：

| 音樂節拍 | 平移 0 秒 | 平移後最低 | 平移後最高 |
|---|---|---|---|
| onsets (original report) | 0.232 | 0.205 | 0.234 |
| librosa beats | 0.284 | 0.177 | 0.284 |
| madmom beats | 0.287 | 0.179 | 0.287 |

## 舞蹈風格（AIST++ 舞風分類器）

| 動作描述 | 4 個樣本的預測 | 命中數 |
|---|---|---|
| The style of the dance is break. | break, break, house, break | 3/4 |
| The genre of the dance is house. | LA style hip-hop, house, house, house | 3/4 |
| The style of the dance is ballet jazz | ballet jazz, ballet jazz, ballet jazz, ballet jazz | 4/4 |
| The style of the dance is street jazz. | street jazz, street jazz, street jazz, street jazz | 4/4 |
| The style of the dance is krump. | krump, krump, krump, krump | 4/4 |
| This is a lock style dance. | lock, lock, lock, lock | 4/4 |
| The style of the dance is middle hip-hop. | LA style hip-hop, middle hip-hop, middle hip-hop, middle hip-hop | 3/4 |
| The style of the dance is pop. | lock, pop, pop, pop | 3/4 |
| The genre of the dance is waack | waack, waack, waack, waack | 4/4 |
| The style of the dance is lock. | lock, waack, pop, LA style hip-hop | 1/4 |
| The genre of the dance is house. | house, house, house, house | 4/4 |
