## 近紅外光譜儀資料分析
Near-infrared Spectroscopy data analysis  
以NIRX公司生產之NIRSCOUNT近紅外光譜儀為資料分析對象

### Preprocessing
基於nirsLAB處理數據的方式, 實現References retrieval、Spike artifacts、Bandpass filter(套用scipy)，並根據Beer-Lambert Law計算oxyHb、deoxyHb。

#### To be continued
1. Removing artifacts from data, 經Artifacts後的區段, triggers應不列入計算
2. Coeffiecient of Variation(CV) > List Good/Bad Channels, 排除超出標準的Channels(& Gain<7)
3. 增加level 1、level 2

#### Package
- 目前各階段均在系統端運行, 考慮系統負荷是否如SPM每個歷程均產生獨立檔案
- tkinter介面轉為PySide2
- 包裝為exe執行檔
