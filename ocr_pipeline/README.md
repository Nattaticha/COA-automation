# OCR Pipeline สำหรับ COA

ใช้คัดเลือกและปรับโมเดล OCR สำหรับ COA กับ RD Spec ทุกขั้นรันบน PC บริษัทแบบ offline และไม่ส่งข้อมูลออกนอกองค์กร
(ต้องต่ออินเทอร์เน็ตแค่ตอนติดตั้งแพ็กเกจและดาวน์โหลดน้ำหนักโมเดลครั้งแรก)

```
PDF ─► ตรวจว่าเป็น Native หรือ Scan ─► แยกโซน (DLA) ─┬─► โซนข้อความ ─► OCR ─► ข้อความ
                                                └─► โซนตาราง  ─► Table reader ─► ตาราง 2 มิติ
```

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.py` | path, DPI, ชื่อโมเดล, สัดส่วน split (แก้ค่าที่นี่ที่เดียว) |
| `preprocess.py` | render หน้า 300 DPI, ลบสีแดง, deskew, denoise, binarize, จำลองภาพสแกน |
| `zoning.py` | **แยกโซนข้อความกับตาราง**: Native ใช้ PyMuPDF / Scan ใช้ PP-DocLayout หรือ OpenCV และอ่านตาราง (grid หรือ paddle_table) |
| `engines.py` | อะแดปเตอร์ Tesseract, EasyOCR, PaddleOCR และ PaddleOCR ที่ fine-tune แล้ว ใช้ interface เดียวกัน |
| `metrics.py` | CER, WER, Cell Accuracy, TEDS และ Precision/Recall ของโซนตาราง |
| `build_dataset.py` | สร้าง ground truth และแบ่ง train/val/test ตาม Supplier (test set ล็อกไว้ใน `split.json`) |
| `compare_models.py` | เทียบโมเดลบน test set ชุดเดียวกัน แล้วเขียนผลลง `results/` |
| `finetune_rec.py` | fine-tune PP-OCRv5 rec บน CPU แล้ว export |
| `extract_coa.py` | รันกับ PDF 1 ไฟล์ ได้ JSON, Excel และรูปกรอบโซนไว้ตรวจด้วยตา |

## ขั้นตอน

**0. ติดตั้ง** (ใน venv `env`)
```powershell
.\env\Scripts\python.exe -m pip install -r requirements-ocr.txt
```
- Tesseract ต้องติดตั้งตัวโปรแกรมแยก (UB Mannheim) และเลือกภาษา Thai ด้วย
  - ถ้าติดตั้งไม่ได้เพราะต้องใช้สิทธิ์ admin ให้ข้าม Tesseract ไป แล้วเทียบ EasyOCR กับ PaddleOCR และ PaddleOCR fine-tune แทน

**1. เตรียมข้อมูล:** คัดลอก COA และ RD Spec ย้อนหลังไปไว้ที่ `data/raw/` โดยคงโครงโฟลเดอร์ `[supplier code]_[ชื่อ]/...` ไว้ เพราะระบบใช้ชื่อโฟลเดอร์แบ่ง split ตาม Supplier

**2. สร้าง dataset**
```powershell
.\env\Scripts\python.exe -m ocr_pipeline.build_dataset
```
- **Native PDF:** ใช้ข้อความในไฟล์เป็นเฉลย ระบบตัดภาพทีละบรรทัด และสร้างภาพจำลองสแกนเพิ่มให้เอง
- **Scan PDF:** ระบบ export หน้าไปไว้ที่ `data/to_label/` ให้ติด label ด้วย PPOCRLabel แล้ววางผลไว้ที่ `data/labeled/<supplier>/Label.txt` จากนั้นรันขั้นนี้ซ้ำ
- `split.json` ถูกสร้างครั้งแรกครั้งเดียว **ห้ามลบ** ถ้าลบ test set จะเปลี่ยนและเทียบผลข้ามรอบไม่ได้

**3. เทียบโมเดล (baseline)**
```powershell
.\env\Scripts\python.exe -m ocr_pipeline.compare_models --engines tesseract,easyocr,paddle --table-readers grid,paddle_table
.\env\Scripts\python.exe -m ocr_pipeline.compare_models --engines paddle --preprocess red,deskew,binarize --note "ablation"
```
- ผลอยู่ที่ `results/<เวลา>/summary.xlsx`
  - `lines_summary`: CER และ WER แยกตามโซน (text/table) และตามชนิดภาพ (native_clean / scan_sim / real_scan)
  - `lines_worst`: บรรทัดที่อ่านผิดมากที่สุด
  - `tables_summary`: Cell Accuracy และ TEDS
  - `layout_summary`: Precision และ Recall ของการหาโซนตาราง
- ทุกรอบที่รันจะถูกต่อท้ายลง `results/model_metrics.csv` พร้อมเวอร์ชันโมเดลและ hash ของ test set

**4. Fine-tune** (1 รอบ บน CPU ใช้เวลาหลายชั่วโมง)
```powershell
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git third_party/PaddleOCR
.\env\Scripts\python.exe -m pip install -r third_party/PaddleOCR/requirements.txt
.\env\Scripts\python.exe -m ocr_pipeline.finetune_rec --epochs 10 --max-train 3000 --run
```
- ระบบเลือกบรรทัดที่มีตัวอักษร `≤ ≥ ± µ ° %` เข้าชุดเทรนก่อน
- ใช้ `val` เลือก checkpoint ส่วน `test` ไม่ถูกใช้เทรน

**5. เทียบหลัง fine-tune**
```powershell
.\env\Scripts\python.exe -m ocr_pipeline.compare_models --engines paddle,paddle_ft
```

**6. ทดลองกับไฟล์จริง**
```powershell
.\env\Scripts\python.exe -m ocr_pipeline.extract_coa "data\raw\...\COA.pdf" --engine paddle_ft
```

## ผลที่เก็บไว้ (อ่านย้อนหลังได้)
| ไฟล์ | เก็บอะไร |
|---|---|
| `results/model_metrics.csv` | ทุกการวัดทุกรอบ แบบ 1 แถวต่อ 1 ค่า (append-only) เปิดด้วย Excel แล้วทำ Pivot ได้ |
| `results/<เวลา>/` | ผลรายบรรทัด (`lines.csv`), รายตาราง (`tables.csv`), `summary.xlsx` และ `run_config.json` ของรอบนั้น |
| `results/finetune_runs.csv` | สรุป fine-tune 1 แถวต่อ 1 รอบ: base model, เวลา, epoch, lr, ไฟล์ train/val, val acc รายรอบ และ best |
| `results/finetune_<run>.json` | รายละเอียด fine-tune ทั้งหมดของรอบนั้น |
| `results/finetune_log.txt` | console log ดิบของการ train ล่าสุด |
| `models/paddle_rec_ft/train/train.log` | log ของ PaddleOCR เอง (ต่อท้ายทุกรอบ) |

บันทึก fine-tune ที่ train เสร็จแล้วเข้า log:
```powershell
.\env\Scripts\python.exe -m ocr_pipeline.log_finetune --note "รอบที่ 1 ..."
```

## ตีความผล
- **CER** คือ `(S+D+I)/N` ระดับตัวอักษร เป็นตัวชี้วัดหลักของตัวเลขและ Lot และควรดูแยกระหว่างโซนตารางกับโซนข้อความ
- **WER** ใช้ได้เฉพาะข้อความภาษาอังกฤษหรือตัวเลข เพราะภาษาไทยไม่เว้นวรรคระหว่างคำ
- **TEDS** = 1 แปลว่าโครงสร้างและข้อความของตารางตรงกันทั้งหมด **Cell Accuracy** คือ % ของเซลล์ที่ค่าตรงตำแหน่ง
- `scan_sim` เป็นภาพจำลอง ก่อนสรุปผลต้องยืนยันกับ `real_scan` ที่ติด label จริงด้วย
- **Field Accuracy** (10 ฟิลด์แบบ end-to-end) จะวัดได้หลังทำขั้น Structuring (Regex) เสร็จ โดยใช้ SAP export เป็นเฉลย
