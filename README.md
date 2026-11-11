# ระบบแนะนำศัลยกรรมและหัตถการจากภาพถ่ายใบหน้า

ผู้ใช้ถ่ายภาพหน้าตรงและด้านข้างผ่านหน้าเว็บ ระบบประเมินกราม คาง ใต้ตา และจมูกทีละบริเวณ แล้วแนะนำหัตถการตามเกณฑ์จากงานวิจัย ผลไม่ใช่การวินิจฉัย การตัดสินใจทำหัตถการต้องผ่านการตรวจโดยแพทย์ api และ web รันบนเครื่องเดียวที่ `127.0.0.1`

- [docs/model.md](docs/model.md) เกณฑ์ กฎแนะนำ ข้อมูล การเตรียมโมเดล และการประเมินผล
- [docs/api.md](docs/api.md) Python API ที่รับภาพแล้วเรียกใช้สิ่งที่ส่วน model ส่งมอบ
- [docs/web.md](docs/web.md) หน้าเว็บที่ถ่ายภาพ คุมท่า และแสดงผล
- [model/reports/summary.md](model/reports/summary.md) สรุปผลทั้งโปรเจกต์

เลขอ้างอิง \[n\] ใช้ชุดเดียวกันทั้งสามไฟล์ใน `docs/` ทุกคำสั่งใน README รันจาก root ของโปรเจกต์ และใช้ได้ทั้ง macOS และ Windows เว้นแต่จะแยกไว้

repo มีโมเดลที่เทรนแล้วอยู่ใน `api/bundle/` ถ้าจะใช้งานอย่างเดียว ให้ทำแค่หัวข้อ "ติดตั้ง" กับ "เปิดใช้งาน" ไม่ต้องโหลดชุดข้อมูลหรือเทรนเอง ส่วนหัวข้อ "เตรียมข้อมูลและเทรน" ใช้เมื่อจะเทรนใหม่

## เงื่อนไขการใช้

น้ำหนักใน `api/bundle/` เทรนจาก SCUT-FBP5500 และ FFHQ กับป้าย FFHQ-Aging ซึ่งทุกชุดใช้ได้เฉพาะงานวิจัยที่ไม่ใช่เชิงพาณิชย์ และ FFHQ กับป้าย FFHQ-Aging ใช้สัญญาอนุญาต CC BY-NC-SA 4.0 (model.md หัวข้อ 5.1) bundle จึงใช้ได้ภายใต้เงื่อนไขเดียวกัน ห้ามใช้เชิงพาณิชย์ repo นี้ไม่มีภาพจากชุดข้อมูล รายงานใน `model/reports/` มีเฉพาะตัวเลขและกราฟ

## เครื่องที่ใช้ได้

- macOS 14 ขึ้นไปบน Apple Silicon หรือ Windows 10/11 แบบ x64 (Windows บน ARM ใช้ไม่ได้ เพราะ torch ไม่มีรุ่นให้)
- ไม่ต้องมี GPU บน Mac ใช้ MPS ส่วน Windows ใช้ torch รุ่น CPU
- Node.js รุ่น LTS ตั้งแต่ 22.12 ขึ้นไป ห้ามใช้รุ่น 23 และ 25 เพราะ Vitest ไม่รองรับ

## ติดตั้ง

### 1. uv, Git และ Node.js

macOS

```bash
brew install uv node
```

Windows (PowerShell) แล้วเปิด terminal ใหม่

```powershell
winget install --id=astral-sh.uv -e
```

```powershell
winget install --id Git.Git -e
```

```powershell
winget install OpenJS.NodeJS.LTS
```

### 2. dependency

```bash
uv sync
```

```bash
npm --prefix web ci
```

uv โหลด Python 3.12 ตาม `.python-version` มาเอง และล็อกรุ่นไว้ใน `uv.lock` สำหรับ macOS บน Apple Silicon กับ Windows x64 เท่านั้น ห้ามเปลี่ยนรุ่นของ `mediapipe` (1.2.0) และ `torch` (2.15.0) เอง เพราะจุดและ feature เปลี่ยนตามรุ่น (api.md หัวข้อ 5)

ตรวจการติดตั้ง

```bash
uv run python -c "import mediapipe, torch; from facemeasure.device import pick_device; print(mediapipe.__version__, torch.__version__, pick_device())"
```

ต้องได้ `1.2.0 2.15.0` ตามด้วย `mps` บน Mac หรือ `cpu` บน Windows (torch อาจมี `+cpu` ต่อท้าย)

### 3. ไฟล์โมเดล

Face Landmarker กับ Image Segmenter ของ MediaPipe น้ำหนัก DINOv2 ViT-B/14 (ราว 346 MB) และโค้ดของ DINOv2 แบบ zip อยู่ในรายการ `api/assets.lock.json` ตัวไฟล์อยู่ใน `api/assets/` ซึ่งอยู่นอก git

```bash
uv run python -m facemeasure.assets
```

- โหลดเฉพาะไฟล์ที่ยังไม่มี แล้วตรวจขนาดและ sha256 กับ lock ถ้า lock ยังไม่มี sha256 จะบันทึกให้ แล้วให้ commit `api/assets.lock.json`
- Google เผยแพร่ไฟล์ของ MediaPipe ที่ลิงก์ `latest` ไฟล์จึงอาจเปลี่ยน ถ้า sha256 ไม่ตรง คำสั่งจะหยุด ให้คัดลอก `api/assets/` ทั้งโฟลเดอร์จากเครื่องที่สร้าง lock มาแทน
- zip ของ DINOv2 ถูกแตกไว้ที่ `api/assets/dinov2_src/` ห้ามแก้ไฟล์ในโฟลเดอร์นี้
- api ไม่โหลดอะไรเองตอนเริ่มทำงาน

## เปิดใช้งาน

ต้องโหลดไฟล์โมเดลแล้ว ส่วน bundle ที่เทรนแล้วมากับ repo

```bash
npm --prefix web run build
```

```bash
uv run python -m api
```

แล้วเปิด `http://localhost:8000` ดูที่ `http://localhost:8000/health` ว่า `model_bundle` เป็นรุ่นที่ต้องการ และ `stub_heads` เป็นรายการว่าง

- ถ้าขาดไฟล์ api จะบอกคำสั่งที่ต้องรันแล้วจบ host ตายตัวที่ `127.0.0.1` เปลี่ยน port ได้ด้วย `--port 8001`
- ถ้า `web/dist/` ยังไม่มี หรือเก่ากว่าไฟล์ใน `web/src/`, `web/index.html`, `shared/quality.json` หรือ `shared/i18n/` api จะขึ้น `WARNING` ให้ build ใหม่แล้วรีเฟรชหน้าเว็บ ไม่ต้องเปิด api ใหม่
- ถ้า bundle สร้างด้วยรุ่น Python, mediapipe, torch หรือไฟล์โมเดลที่ต่างจากเครื่องนี้ จะขึ้น `WARNING` แต่ยังเปิดได้
- ถ้า Windows ถามสิทธิ์ firewall ไม่ต้องอนุญาต เพราะ api ไม่รับการเชื่อมต่อจากเครื่องอื่น

### กล้อง

- เปิดที่ `http://localhost:8000` หรือ `http://127.0.0.1:8000` เท่านั้น เพราะเบราว์เซอร์ให้ใช้กล้องโดยไม่มี HTTPS เฉพาะ localhost ทั้งสองที่อยู่ต้องอนุญาตกล้องแยกกัน
- ต้องต่ออินเทอร์เน็ตตอนเปิดกล้องครั้งแรก เพื่อโหลด MediaPipe จาก jsDelivr และ `storage.googleapis.com` ฟอนต์โหลดจาก Google Fonts ภาพและค่าวัดไม่ถูกส่งไปที่ใดนอกจาก api บนเครื่องนี้
- ถ้าเบราว์เซอร์ไม่ถามสิทธิ์หรือภาพดำ บน macOS เปิดสวิตช์ของเบราว์เซอร์ใน System Settings > Privacy & Security > Camera แล้วเปิดเบราว์เซอร์ใหม่ บน Windows เปิด "Camera access", "Let apps access your camera" และ "Let desktop apps access your camera" ใน Settings > Privacy & security > Camera และปิดโปรแกรมอื่นที่ใช้กล้องอยู่

### ตัวเลือกใน URL

| ส่วนท้าย URL | ผล |
| --- | --- |
| `?lab=1` | โหมดทดลอง (web.md หัวข้อ 7) แสดงค่าวัดสดระหว่างคุมท่า และในหน้าผลแสดงรหัสใน `warnings`, `landmark_value`, P(เกรด ≥ k) ของใต้ตา ค่าที่เว็บวัดเทียบกับ `quality.frontal` ของ api และ JSON ทั้งหมด |
| `?lang=en` | ภาษาของหน้าเว็บและข้อความในผล ภาษาที่ไม่มีไฟล์ใช้ภาษาไทย |
| `?demo=<ชื่อ>` | เฉพาะ `npm run dev` แสดงหน้าผลจาก `web/tests/fixtures/<ชื่อ>.json` โดยไม่ต้องเปิด api ถ้าไม่ใส่ชื่อจะได้รายชื่อไฟล์ทั้งหมด |

ใช้หลายตัวพร้อมกันได้ เช่น `?lab=1&lang=en`

### กรอบนำท่าที่ระยะ 1 ม.

ทำใหม่ทุกครั้งที่เปลี่ยนกล้อง โดยไม่ต้องบันทึกภาพ (web.md หัวข้อ 9)

1. เปิด `http://localhost:8000/?lab=1` ไปหน้าภาพหน้าตรง เปิดกล้อง 1080p แล้วยืนห่างกล้อง 1 ม. ตามตลับเมตร
2. อ่านค่า `stream`, `iod_px` และ `face_width_px` ใต้ภาพ แล้วดูว่ากรอบพอดีใบหน้าหรือไม่
3. ถ้าไม่พอดี แก้ `GUIDE_FRAME_WIDTH` ใน `web/src/constants.ts` (ความกว้างของศีรษะในกรอบเทียบกับความกว้างของภาพ) แล้ว build ใหม่
4. ยืนยันว่า `iod_px` อยู่ระหว่าง `min_iod_px` (75 px) กับ `train_face_iod_px.max` ของ bundle (110 px) และข้อความ "ถอยห่าง" กับ "เข้าใกล้" ขึ้นตามระยะ แล้วเพิ่มแถวในตาราง

| กล้อง | stream | `iod_px` ที่ 1 ม. | `face_width_px` ที่ 1 ม. | `GUIDE_FRAME_WIDTH` | วันที่วัด |
| --- | --- | --- | --- | --- | --- |
| กล้องในตัวของ Mac M2 | 1080×1920 | 78.4 | 184.9 | 0.13 (ไม่เปลี่ยน) | 2026-11-09 |

กับกล้องนี้ กรอบกว้าง 217.5 px ตามขั้นต่ำ 2.9 × 75 px เพราะ 0.13 × 1080 = 140.4 px แคบกว่า ใบหน้าที่ 1 ม. กว้าง 85% ของกรอบ ระยะที่ผ่านทั้งสองข้อคือราว 0.85–1.04 ม. ถ้าใช้กล้องแนวนอน 1920×1080 กรอบจะกว้าง 249.6 px ต้องวัดซ้ำ

## api

FastAPI ตาม `docs/api.md` รับภาพหน้าตรง ภาพด้านข้าง และ `meta` ในการเรียกครั้งเดียว ไม่เก็บภาพ ใช้ได้กับ bundle ทุกรุ่นตั้งแต่ stub

```bash
uv run python -m api
```

หรือเรียก uvicorn เอง ซึ่งต้องเรียก `api.main` จาก root เพราะโค้ดใน `api/` import แบบสัมพัทธ์

```bash
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

### ลองเรียก

macOS

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/analyze -F frontal=@frontal.jpg -F profile=@profile.jpg -F 'meta={"age": 22, "sex": "female", "frontal_source": "camera"}'
```

Windows (PowerShell) ใช้ `curl.exe` และเขียน `meta` ลงไฟล์ก่อน เพราะ PowerShell บางรุ่นตัดเครื่องหมายคำพูดในอาร์กิวเมนต์

```powershell
curl.exe http://127.0.0.1:8000/health
```

```powershell
Set-Content -Path meta.json -Value '{"age": 22, "sex": "female", "frontal_source": "camera"}' -Encoding ascii
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze -F frontal=@frontal.jpg -F profile=@profile.jpg -F "meta=<meta.json"
```

หรือใช้ `api/try_api.py` ซึ่งพิมพ์สถานะ เกรด ค่าวัด รหัส และหัตถการของแต่ละบริเวณ โดยไม่บันทึกภาพหรือผล

```bash
uv run python api/try_api.py frontal.jpg profile.jpg --age 22 --sex female
```

ตัวเลือกคือ `--lab`, `--upload` (ภาพหน้าตรงเลือกจากเครื่อง), `--lang`, `--json` (พิมพ์ผลทั้งหมด) และ `--port` ถ้าไม่ใส่ภาพด้านข้าง คางและสันจมูกจะได้ `missing_input` ลองกับภาพจากชุดข้อมูลเท่านั้น

### รายละเอียดที่ควรรู้

- `meta` ตรวจเข้ม: `age` ต้องเป็นจำนวนเต็มของ JSON 0–120 (`22.0`, `"22"` และ `true` ไม่ผ่าน) `sex` เป็น `female` หรือ `male` ส่วน `frontal_source` (`camera`), `lab` (`false`) และ `lang` (`th`) มีค่าเริ่มต้น ฟิลด์อื่นไม่รับ และไฟล์ `profile` ที่ว่างถือว่าไม่ได้ส่ง
- ข้อผิดพลาดมี body เป็น `{"error": รหัส, "request_id": ...}`

| สถานะ | รหัส | เมื่อ |
| --- | --- | --- |
| 422 | `invalid_request` | ไม่มี `frontal` หรือ `meta`, `meta` ผิดรูปแบบ หรือ body ไม่ใช่ multipart |
| 415 | `unsupported_media_type` | ไฟล์ไม่ใช่ JPEG หรือ PNG (ดูจากไบต์แรก) |
| 400 | `decode_failed` | JPEG หรือ PNG ที่ถอดรหัสไม่ได้ |
| 413 | `too_large` | คำขอเกิน `request.max_bytes` ของ `shared/quality.json` (25 MB) หรือภาพเกิน 50 ล้านพิกเซล |
| 500 | `internal_error` | ผิดพลาดภายใน |

- ประมวลผลทีละคำขอ log มีหนึ่งบรรทัดต่อคำขอ คือ `request_id`, `status`, `duration_ms` และ `error` ไม่มีภาพ `meta` หรือค่าวัด จึงไม่มี traceback ของข้อผิดพลาด 500 ถ้าต้องไล่ ให้เรียกซ้ำผ่าน `TestClient` ในเทสต์
- บริเวณที่หัวยังเป็น stub ได้ `not_assessable` ด้วยรหัส `model_not_ready` (โหมดทดลองใส่ไว้ใน `warnings`)
- `/docs` และ `/redoc` ปิดไว้เพราะโหลดไฟล์จาก CDN ส่วน `/openapi.json` เปิดอยู่

### ตรวจกับโมเดลจริง

```bash
uv run python model/scripts/api_check.py
```

เรียก api ด้วยโมเดลจริงและภาพจาก SCUT-FBP5500, ภาพที่เกรดแล้ว และ FEI ตรวจความตรงกับตอนเทรน การส่งซ้ำ การหมุนด้วย EXIF การกลับภาพด้านข้าง รหัสคุณภาพภาพ โหมดทดลอง และเวลาต่อคำขอ (api.md หัวข้อ 7) แล้วเขียน `model/reports/api_check.md` ใช้เวลาราว 1 นาทีบน Mac ใส่ `--only exif timing` เพื่อรันเฉพาะบางข้อ ถ้ามีข้อที่ไม่ผ่าน คำสั่งจะจบด้วยรหัสผิดพลาด

## ชุดข้อมูล

ใช้เฉพาะตอนเตรียมข้อมูลและเทรนใหม่ ข้อมูลทั้งหมดอยู่ใน `data/` ซึ่งอยู่นอก git ใช้ได้เฉพาะชุดข้อมูลใน model.md หัวข้อ 5.1 SCUT-FBP5500 ต้องโหลดเองจาก <https://github.com/HCIILAB/SCUT-FBP5500-Database-Release> ส่วน FFHQ และ FEI สคริปต์โหลดให้

- ห้าม commit ภาพหรือไฟล์จากชุดข้อมูล ภาพที่มีใบหน้าอยู่ใน `data/outputs/` เท่านั้น ส่วนรายงานใน `model/reports/` มีเฉพาะตัวเลขและกราฟ จึง commit ได้
- ภาพ FEI ใช้ได้เฉพาะงานวิจัย ห้ามแจกจ่าย
- ห้ามลบหรือแก้ไฟล์ใน `data/raw/` สคริปต์ที่โหลดข้อมูลไม่เขียนทับไฟล์ที่มีอยู่ ถ้าเจอไฟล์ที่ไม่ตรงจะบอกให้ลบเอง

```text
data/
  raw/
    SCUT-FBP5500_v2/     วางเอง ต้องมี Images/ และ facial landmark/ อย่างละ 5,500 ไฟล์ และ train_test_files/
    ffhq/                grading_set.py โหลดเฉพาะภาพที่ใช้และป้ายของ FFHQ-Aging
    fei/                 profile_check.py โหลดและแตกภาพไว้ที่ images/
    wu2021_fig2/         วางภาพ A.png ถึง H.png จากรูปที่ 2 ของ [3] เอง ใช้ตรวจหัวกรามเท่านั้น
  labels/                label และผลวัดทีละภาพ เช่น scut_labels.csv, undereye_labels.csv, frontal_mediapipe.csv, fei_profile.csv
  grading/               ชุดภาพเกรดใต้ตา: manifest.csv (ห้ามแชร์), share/ ที่ส่งให้ผู้ให้เกรด และ results/ ที่ได้กลับมา
  features/              feature ของ DINOv2, จุดของ MediaPipe และผลที่เก็บไว้ทำต่อจากเดิม
  outputs/               ภาพตรวจที่มีใบหน้า และผลระหว่างทางของการเทรน (ratio_heads/, undereye_head/)
```

ถ้าดึง feature เครื่องหนึ่งแล้วเทรนอีกเครื่อง ให้คัดลอก `data/features/` ไปทั้งโฟลเดอร์

## เตรียมข้อมูลและเทรน

รันตามลำดับหัวข้อ สคริปต์ที่ใช้เวลานานมีแถบ progress ถ้าหยุดกลางทาง รันคำสั่งเดิมจะทำต่อจากเดิม และ `--restart` เริ่มใหม่ทั้งหมด สคริปต์ที่เขียนภาพตรวจรับ `--no-check-images` ถ้าไม่ต้องการภาพตรวจ

### 1. ชุดภาพสำหรับเกรดใต้ตา

เตรียมภาพตาม model.md หัวข้อ 5.2 และ 5.3

```bash
uv run python model/scripts/grading_set.py select
```

```bash
uv run python model/scripts/grading_set.py download
```

```bash
uv run python model/scripts/grading_set.py check
```

```bash
uv run python model/scripts/grading_set.py package
```

- `select` โหลดป้าย FFHQ-Aging กับ `ffhq-dataset-v2.json` (255 MB) แล้วเขียน `data/grading/manifest.csv` ถ้าโหลดไฟล์ json ไม่ได้ ให้เปิด <https://drive.google.com/open?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA> ในเบราว์เซอร์ที่ล็อกอิน Google แล้ววางไว้ที่ `data/raw/ffhq/`
- `download` โหลดภาพ FFHQ ราว 1,100 ภาพจาก Google Drive ถ้า Drive แจ้งโควต้าเต็ม ให้รันใหม่หลังราว 24 ชั่วโมง
- `check` ตรวจภาพ แทนภาพขาวดำด้วยภาพสำรอง และทำภาพย่อไว้ที่ `data/outputs/grading_contact/` ดูภาพย่อแล้วเปลี่ยนภาพที่ใช้ไม่ได้ เช่น มีมือหรือผมบังใต้ตา ใส่ฟิลเตอร์หนัก หรือเบลอ แล้วรัน `check` อีกครั้ง ภาพที่โหลดไม่ได้ซ้ำหลายครั้งก็เปลี่ยนแบบเดียวกันด้วย `--reason "download failed"`

  ```bash
  uv run python model/scripts/grading_set.py replace 00123.png AF45.jpg --reason "hand covers eye"
  ```

- `package` สร้าง `practice/`, `pilot_200/` และ `full_600/` พร้อม zip ใน `data/grading/share/` แต่ละโฟลเดอร์มี `undereye_grader.html` กับ `README.txt` สำหรับผู้ให้เกรด
- `select` ไม่เขียนทับ manifest ที่เคยเปลี่ยนภาพแล้ว เว้นแต่ใส่ `--force` ซึ่งจะลบประวัติการเปลี่ยนภาพ

ผู้ให้เกรดฝึกด้วย `practice/` เกรด `pilot_200/` เพื่อดูสัดส่วนของเกรด แล้วเกรดให้ครบใน `full_600/` จากนั้นวาง CSV ที่ส่งออกจากเครื่องมือไว้ที่ `data/grading/results/` ถ้ามีผู้ให้เกรดหลายคน ให้ทุกคนเกรดทุกภาพ ดู kappa ในแท็บ "สรุปผลหลายคน" และเกรดข้างที่ตกลงร่วมกันด้วยชื่อ `consensus`

### 2. label ของ SCUT-FBP5500

คำนวณอัตราส่วนกรามและฐานจมูกจากจุด 86 จุด แล้วแบ่งชุด train/val/test ด้วย seed คงที่ ใช้เวลาไม่ถึงนาที

```bash
uv run python model/scripts/make_scut_labels.py
```

ได้ `data/labels/scut_labels.csv`, `scut_skipped.csv` (ไฟล์ที่ไม่มี label พร้อมเหตุ), `model/reports/scut_labels.md` และภาพตรวจใน `data/outputs/scut_label_check/` ถ้าต้องการดูจุดของ SCUT ทีละภาพ ให้ใช้เครื่องมือนี้ และใส่ `--out` ไว้ใน `data/outputs/` เสมอ เพราะค่าเริ่มต้นคือโฟลเดอร์ `preview/` ที่ตำแหน่งที่รัน

```bash
uv run python tools/scut_landmark_preview.py --images "data/raw/SCUT-FBP5500_v2/Images" --pts "data/raw/SCUT-FBP5500_v2/facial landmark" --out data/outputs/scut_preview
```

### 3. ตรวจภาพหน้าตรง

รัน `analyze_frontal` กับ SCUT-FBP5500 ทุกภาพและ FFHQ ที่คัดไว้ รวมราว 6,000 ภาพ ใช้เวลาราว 20 นาทีบน CPU

```bash
uv run python model/scripts/frontal_check.py
```

ได้ `data/labels/frontal_mediapipe.csv`, `data/features/landmarks_frontal.npz`, `model/reports/frontal_mediapipe.md` และภาพตรวจใน `data/outputs/frontal_check/`

### 4. ตรวจภาพด้านข้างกับ FEI

```bash
uv run python model/scripts/profile_check.py download
```

```bash
uv run python model/scripts/profile_check.py run
```

- `download` โหลด zip สี่ไฟล์ (ราว 354 MB) จากเว็บของ FEI บันทึก sha256 แตกภาพไว้ที่ `data/raw/fei/images/` และทำแผ่นรวมเลขท่า `data/outputs/fei_numbering.jpg` ถ้าเว็บของ FEI โหลดไม่ได้ ให้รันใหม่ภายหลัง ห้ามใช้แหล่งอื่นโดยไม่ได้ตกลงกันก่อน
- `run` รัน `analyze_profile` กับ 1,200 ภาพ ใช้เวลาราว 10 นาทีบน CPU แล้วเขียน `model/reports/fei_profiles.md` และภาพตรวจใน `data/outputs/fei_profile_check/`

### 5. bundle ชั่วคราว

ใช้เฉพาะตอนพัฒนาก่อนเทรนหัวเสร็จ คำสั่งนี้เขียน `api/bundle/` รุ่น `0.0.0-stub` ทับ bundle ที่เทรนแล้วของ repo กู้คืนได้ด้วย `git restore api/bundle` ห้าม commit stub เพราะ `tests/test_repo_privacy.py` จะไม่ผ่าน

```bash
uv run python model/scripts/make_stub_bundle.py
```

หัวทั้งสามทายค่าเดิมทุกภาพ คือมัธยฐานของ `scut_labels.csv` สำหรับกรามและฐานจมูก และ P(เกรด ≥ 1) = 0.5, P(เกรด ≥ 2) = 0.2 สำหรับใต้ตา หัวข้อ 7 และ 8 จะเขียนรุ่น `0.1.0` และ `0.2.0` ทับโฟลเดอร์เดิม

### 6. feature ของ DINOv2

ย่อภาพของชุดข้อมูลให้เหมือนภาพจากเว็บแคมผ่านเส้นทางเดียวกับ api แล้วดึง feature ไว้ให้เทรนหัว (model.md หัวข้อ 6) ต้องมี `scut_labels.csv` กับ `manifest.csv` ก่อน

```bash
uv run python model/scripts/extract_features.py face
```

```bash
uv run python model/scripts/extract_features.py undereye
```

```bash
uv run python model/scripts/extract_features.py report
```

- `face` ทำครอปทั้งใบหน้าของ SCUT-FBP5500 รวม 16,598 ครอป ใช้เวลาราว 60–80 นาทีบน MacBook Air M2
- `undereye` ทำครอปใต้ตาของภาพที่จะเกรด รวม 6,184 ครอป ใช้เวลาราว 15–20 นาที ไม่ต้องรอผลเกรด
- `report` เขียน `model/reports/features.md` และภาพตรวจใน `data/outputs/features_check/`
- ถ้าเปลี่ยนค่าใน `facemeasure/degrade.py` ต้องเพิ่ม `DEGRADE_VERSION` แล้วดึง feature ใหม่

### 7. หัวกรามและหัวฐานจมูก

ridge regression บน feature ของครอปทั้งใบหน้า (model.md หัวข้อ 4.1 และ 7.1) ก่อน `wu2021` ต้องวางภาพไว้ใน `data/raw/wu2021_fig2/` ภาพชุดนี้ห้ามใช้เทรนและห้ามใส่ในรายงาน

```bash
uv run python model/scripts/train_ratio_heads.py select
```

```bash
uv run python model/scripts/train_ratio_heads.py evaluate
```

```bash
uv run python model/scripts/train_ratio_heads.py wu2021
```

```bash
uv run python model/scripts/train_ratio_heads.py bundle
```

- `select` เลือกชนิด feature กับ alpha บนชุด val ใช้เวลาราว 4 นาที และหน่วยความจำสูงสุดราว 4.3 GB เครื่องที่มี RAM 8 GB อาจไม่พอ
- `evaluate` ทายชุด test ครั้งเดียว เทียบกับ baseline จากจุดของ MediaPipe และตั้งช่วงก้ำกึ่ง ถ้าเข้าเงื่อนไขหยุดจะพิมพ์ `FLAG` และ `bundle` จะไม่เขียนจนกว่าจะคุยกันแล้วสั่ง `bundle --accept-flags`
- `wu2021` ตรวจหัวกรามกับภาพ 8 ภาพของ \[3\] (model.md หัวข้อ 3.1)
- `bundle` เขียน `api/bundle/` รุ่น `0.1.0` แล้วตรวจว่า api ได้ค่าตรงกับตอนเทรน ถ้าไม่ผ่านจะจบด้วยรหัสผิดพลาด
- ทุกคำสั่งเขียน `model/reports/jaw_nose_heads.md` ใหม่ ผลระหว่างทางอยู่ใน `data/outputs/ratio_heads/`

### 8. หัวใต้ตา

หัว ordinal แบบ CORAL บน feature ของครอปใต้ตา ปรับเทียบด้วย temperature scaling (model.md หัวข้อ 4.2 และ 7.1) ต้องมี bundle รุ่น `0.1.0` และ CSV ของชุด `full_600` ใน `data/grading/results/` ก่อน

```bash
uv run python model/scripts/train_undereye_head.py labels
```

```bash
uv run python model/scripts/train_undereye_head.py select
```

```bash
uv run python model/scripts/train_undereye_head.py calibrate
```

```bash
uv run python model/scripts/train_undereye_head.py evaluate
```

```bash
uv run python model/scripts/train_undereye_head.py cross
```

```bash
uv run python model/scripts/train_undereye_head.py bundle
```

- `labels` รวม CSV ของผู้ให้เกรดเป็น `data/labels/undereye_labels.csv` และวัดความสอดคล้องเมื่อมีผู้ให้เกรดตั้งแต่สองคน
- `select` เลือกชนิด feature กับ λ บนชุด val ใช้เวลาราว 5 นาที `calibrate` หา T บนชุด val
- `evaluate` ทายชุด test ครั้งเดียว `cross` เทรนด้วย FFHQ อย่างเดียวแล้วตรวจกับ SCUT (ไม่ลง bundle) ต้องรันหลัง `evaluate`
- `bundle` เขียน `api/bundle/` รุ่น `0.2.0` โดยคงหัวกรามและฐานจมูกไว้ แล้วตรวจว่า api ได้ค่าตรงกับตอนเทรน เมื่อเทรนใหม่แล้วให้ commit `api/bundle/` พร้อมรายงาน
- ทุกคำสั่งเขียน `model/reports/undereye_head.md` ใหม่ โดยใช้รหัส R1, R2, … แทนชื่อผู้ให้เกรด ชื่อจริงอยู่ใน `data/outputs/undereye_head/raters.json` ซึ่งห้ามแชร์

## พัฒนาหน้าเว็บ

Vite กับ TypeScript โดยไม่ใช้ framework ตาม `docs/web.md` คำสั่งในหัวข้อนี้รันใน `web/`

### dev server

เปิด api ไว้อีกหน้าต่าง แล้วเปิด `http://127.0.0.1:5173`

```bash
npm run dev
```

Vite ส่งต่อ `/analyze` และ `/health` ไปที่ api และฟังเฉพาะพอร์ต 5173 ถ้าพอร์ตนี้ถูกใช้อยู่จะจบพร้อมข้อความ

### ผลตัวอย่างและโหมด demo

เทสต์ของหน้าเว็บกับ `?demo` ใช้ผลตัวอย่างใน `web/tests/fixtures/` ที่สร้างจากค่าวัดสังเคราะห์ ไม่มีภาพ สร้างใหม่ทุกครั้งที่แก้ข้อความใน `th.json` ที่ไม่ใช่ `ui.`, เกณฑ์, กฎ หรือ `response.py` (รันที่ root) ถ้าไฟล์ที่ commit ไว้ไม่ตรงกับที่สร้างใหม่ `tests/test_make_web_fixtures.py` จะไม่ผ่าน ภาพของทุกกรณีเป็นแนวตั้ง 1280×1600 ยกเว้น `webcam_1080p` ที่เป็นภาพเว็บแคม 1920×1080 ซึ่งใบหน้าเล็กเท่าภาพจริง ใช้ดูว่าเส้นวัดเห็นชัดแค่ไหน (`?demo=webcam_1080p`)

```bash
uv run python model/scripts/make_web_fixtures.py
```

### เทสต์

```bash
npm run typecheck
```

```bash
npm run test
```

```bash
npm run build
```

Vitest ไม่ต้องเปิดเบราว์เซอร์หรือ api กล้องกับ MediaPipe ใช้ตัวจำลอง

### เทสต์ทั้งระบบด้วยกล้องจำลอง (ใช้หรือไม่ก็ได้)

Playwright เปิด Chromium ที่มีกล้องจำลองซึ่งเล่นภาพ test ของ SCUT-FBP5500 แล้วทำหนึ่งรอบจริงกับ api ทั้งภาษาไทยและภาษาอังกฤษ ตรวจการถ่ายอัตโนมัติ การ์ด storage คำขอเครือข่าย console ภาพที่หมุนด้วย EXIF และการเข้าถึง (`web/e2e/audit.ts`) ต้องรันหัวข้อ 3 และ 4 ของ "เตรียมข้อมูลและเทรน" มาก่อน และต้องต่ออินเทอร์เน็ต

1. ครั้งแรกครั้งเดียว โหลด Chromium ของ Playwright (หลายร้อย MB)

   ```bash
   npx playwright install chromium
   ```

2. ที่ root สร้างวิดีโอจำลองใน `data/outputs/e2e/` ซึ่งห้าม commit เพราะมีใบหน้า

   ```bash
   uv run python model/scripts/make_e2e_video.py
   ```

3. ปิด api ที่เปิดอยู่ เพราะเทสต์เปิด api เองที่ port 8000 แล้ว build และรันเทสต์

   ```bash
   npm run build
   ```

   ```bash
   npm run e2e
   ```

ถ้าไม่มี Chromium หรือวิดีโอจำลอง เทสต์จะ skip ผลอยู่ใน `data/outputs/e2e/test-results/` โดยไม่มีภาพหน้าจอ วิดีโอ หรือ trace

### ถ่ายภาพหน้าจอ

ถ่ายภาพทุกหน้าจอเป็นภาษาไทย ไม่ใช้โหมดทดลอง และไม่มีใบหน้าในภาพ หน้าผลใช้โหมด demo ที่วาดเส้นวัดบนภาพพื้นสีเรียบ ส่วนกล้องจำลองกับภาพที่เลือกจากเครื่องก็เป็นภาพสีเรียบ ไม่ต้องเปิด api หรือมีไฟล์โมเดล แต่ต้องโหลด Chromium ของ Playwright ก่อน (ขั้นที่ 1 ของหัวข้อก่อนหน้า) และต้องต่ออินเทอร์เน็ตเพื่อโหลดฟอนต์และ MediaPipe

```bash
npm run screenshots
```

ได้ภาพ PNG กว้าง 1440 px ที่ความละเอียด 2 เท่าแบบเต็มหน้า ใน `data/outputs/ui_screenshots/` (นอก git) ชื่อไฟล์เรียงตามลำดับการใช้งาน คือหน้าแรก หน้าแรกตอนต่อ api ไม่ได้ ภาพหน้าตรงและภาพด้านข้างตอนกล้องปิดและเปิด หน้าตรวจภาพ แล้วหน้าผลของผลตัวอย่างทุกกรณีที่ไม่ใช่โหมดทดลอง (`08_results_<ชื่อ>.png` และ `_procedures` แบบกางหัตถการ) ภาพเก่าในโฟลเดอร์ถูกลบทุกครั้งที่รัน เครื่องมือเปิด dev server เองหรือใช้ตัวที่เปิดอยู่ ถ้าเพิ่มหน้าจอ ให้แก้รายการใน `web/screenshots/shots.ts`

## ข้อความและการเพิ่มภาษา

ข้อความที่ผู้ใช้เห็นทั้งหมดอยู่ใน `shared/i18n/` ซึ่ง api และ web ใช้ร่วมกัน `th.json` เป็นภาษาหลัก และ `en.json` มีครบทุกคีย์ คีย์ที่ขึ้นต้นด้วย `ui.` เป็นข้อความบนหน้าเว็บ ส่วนคีย์อื่นเป็นข้อความในผลของ api ถ้าแก้ถ้อยคำ ให้แก้คีย์เดียวกันทั้งสองไฟล์แล้วรันเทสต์ ถ้าแก้คีย์ที่ไม่ใช่ `ui.` ให้สร้างผลตัวอย่างใหม่ด้วย

เพิ่มภาษาได้โดยไม่ต้องแก้โค้ด (web.md หัวข้อ 8.4)

1. สร้าง `shared/i18n/<รหัสภาษา>.json` เช่น `ja.json` ด้วยคีย์ชุดเดียวกับ `th.json` คีย์ที่ยังไม่แปลจะใช้ภาษาไทย ตัวแทนค่าอย่าง `{low}` ต้องเหมือนของภาษาไทย และห้ามมีคีย์ที่ `th.json` ไม่มี
2. รัน `uv run pytest tests/test_i18n.py` และ `npm --prefix web run test`
3. build หน้าเว็บใหม่ เพราะหน้าเว็บรวมไฟล์ภาษาไว้ตอน build แล้วเปิดด้วย `?lang=ja` หรือเลือกจากตัวเลือกภาษา

## path และอุปกรณ์

- path ทั้งหมดอยู่ใน `config.toml` แบบสัมพัทธ์กับ root โค้ดอ่านผ่าน `facemeasure.paths` เช่น `paths.path("scut")` root คือโฟลเดอร์ที่มี `config.toml` หรือกำหนดเองด้วยตัวแปร `CAPSTONE_ROOT`
- `facemeasure.device.pick_device()` เลือก CUDA, MPS แล้ว CPU ตามลำดับ บังคับได้ด้วยตัวแปร `FACEMEASURE_DEVICE` (`auto`, `cpu`, `mps`, `cuda` หรือ `cuda:N`) ถ้าเปลี่ยนรุ่น torch แล้วเจอ op ที่ MPS ไม่รองรับ ให้บังคับใช้ CPU ก่อนแล้วแจ้งทีม

macOS

```bash
FACEMEASURE_DEVICE=cpu uv run pytest
```

Windows (PowerShell)

```powershell
$env:FACEMEASURE_DEVICE = "cpu"; uv run pytest
```

## รันเทสต์

```bash
uv run pytest
```

```bash
uv run ruff check .
```

เทสต์ที่ต้องใช้ข้อมูลจริง ไฟล์โมเดล หรือ feature จะ skip เมื่อยังไม่มีไฟล์ `tests/test_repo_privacy.py` ตรวจว่าใน git ไม่มีภาพนอกจากกราฟใน `model/reports/figures/` ไม่มีไฟล์จาก `data/` หรือ `api/assets/` และ bundle ใน git ไม่ใช่ stub ส่วน `tests/test_doc_refs.py` ตรวจว่าการอ้างหัวข้อของ docs ทุกที่ชี้ไปที่หัวข้อที่มีอยู่จริง เทสต์ของหน้าเว็บอยู่ในหัวข้อ "พัฒนาหน้าเว็บ"

## โครงโฟลเดอร์

```text
capstone-project/
  config.toml            path ทั้งหมด
  pyproject.toml, uv.lock, .python-version
  docs/                  สเปก model, api และ web
  shared/                ไฟล์ที่ api กับ web ใช้ร่วมกัน: quality.json (เกณฑ์คุณภาพภาพ), i18n/ (ไฟล์ภาษา), testdata/
  src/facemeasure/       โค้ดวัดค่า ที่เดียวที่ model และ api เรียกใช้
    frontal.py           analyze_frontal ทางเข้าเดียวของภาพหน้าตรง
    profile.py           analyze_profile ทางเข้าเดียวของภาพด้านข้าง
    model_input.py       analyze_for_model ทางเข้าของ api: ย่อภาพที่ใบหน้าใหญ่กว่าช่วงที่เทรนก่อนครอป
    landmarks.py, mp_points.py, pose.py, quality.py, crops.py, face_geometry.py, scut.py, fei.py
                         MediaPipe, เลขจุด, ท่าศีรษะ, ค่าคุณภาพ, ครอป, อัตราส่วน และไฟล์ของชุดข้อมูล
    codes.py, criteria.py, undereye.py, rules.py, response.py
                         รหัสในผล, เส้นแบ่งเกรด, เกรดใต้ตา, กฎแนะนำ และการจัดผลตาม api.md หัวข้อ 2.2
    bundle.py, backbone.py, degrade.py
                         api/bundle/, DINOv2 และการย่อภาพเลียนแบบเว็บแคม
    assets.py, paths.py, device.py, image_io.py, i18n.py, draw.py
    training/            โค้ดเทรนที่ api ไม่เรียก
  model/
    scripts/             สคริปต์เตรียมข้อมูล เทรน และตรวจ (เทสต์ import ได้ด้วยชื่อไฟล์)
    reports/             รายงานที่ commit ได้ มีเฉพาะตัวเลขและกราฟ
  tools/                 undereye_grader.html (เครื่องมือเกรดใต้ตา) และ scut_landmark_preview.py
  api/                   FastAPI: main.py, engine.py, schemas.py, try_api.py และ assets.lock.json
    assets/              ไฟล์โมเดลที่ดาวน์โหลดมา (นอก git)
    bundle/              สิ่งที่ส่วน model ส่งมอบ: bundle ที่เทรนแล้ว
  web/                   หน้าเว็บ Vite + TypeScript
    src/                 app.ts, state.ts, api.ts, guidance.ts (MediaPipe และการตรวจท่า), constants.ts ฯลฯ
      screens/           หน้าจอแต่ละหน้า และกรอบนำท่า guide_layer.ts
      results/           เส้นวัด ภาพรวม การ์ด และส่วนของโหมดทดลอง
      styles/            tokens.css และ components.css
    build/csp.ts         Content-Security-Policy ที่ใส่ตอน build
    tests/               เทสต์ Vitest และ fixtures/
    e2e/                 เทสต์ทั้งระบบด้วย Playwright
    screenshots/         ถ่ายภาพหน้าจอสำหรับรายงานด้วย Playwright
  tests/                 เทสต์ pytest
  data/                  ชุดข้อมูลและผล (นอก git)
```
