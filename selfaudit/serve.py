"""A functional local web UI: ``selfaudit-serve`` (or ``python -m selfaudit.serve``).

Open a browser, drop a CSV (or paste a URL, or pick a free live source), and get
the trust verdict + the full report rendered live — interactively, in the
browser. Pure stdlib ``http.server``: no web framework, no build step, and your
data never leaves your machine (the server binds to localhost).
"""

from __future__ import annotations

import base64
import json
import webbrowser
from collections.abc import Callable
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer

from .datasets import Dataset, infer_checks, parse_text, parse_xlsx_bytes, svg_chart
from .datasetscanner import SelfAuditingDatasetScanner
from .sources import SourceUnavailable, crypto_prices, fetch_csv, open_meteo, usgs_earthquakes

_SOURCES: dict[str, Callable[..., Dataset]] = {
    "open-meteo": open_meteo,
    "usgs": usgs_earthquakes,
    "crypto": crypto_prices,
}


def _sample_csv() -> str:
    """A tiny, relatable demo table (customers/orders) with a planted gap and an
    outlier, so the 'Sample CSV' button shows a non-trivial verdict with zero
    setup — domain-neutral, not sensor-specific."""
    rows = ["customer_id,age,order_amount,status"]
    statuses = ("active", "churned")
    for i in range(12):
        age = 25 + (i * 3) % 40
        amount = "" if i == 8 else ("99999" if i == 5 else f"{40 + (i * 7) % 120}")
        rows.append(f"C{i + 1:03d},{age},{amount},{statuses[i % 2]}")
    return "\n".join(rows) + "\n"


_SAMPLE_CSV = _sample_csv()

_INDEX = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>selfaudit</title><style>
body{font:15px/1.6 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px}h1{margin:0 0 2px}
.sub{color:#656d76;margin:0 0 18px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:18px;margin-bottom:16px}
.drop{border:2px dashed #c7d0d9;border-radius:10px;padding:28px;text-align:center;color:#57606a;
cursor:pointer}.drop.over{border-color:#1a7f37;background:#f2fbf5}
.row{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
input[type=text]{flex:1;min-width:220px;padding:9px 11px;border:1px solid #d0d7de;border-radius:8px}
select,button{padding:9px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fff}
button{background:#1f883d;color:#fff;border:0;font-weight:700;cursor:pointer}
iframe{width:100%;height:620px;border:1px solid #d0d7de;border-radius:12px;background:#fff}
.muted{color:#8c959f;font-size:13px}
.examples{margin-top:14px;color:#57606a;font-size:14px}
button.secondary{background:#f3f5f7;color:#1f2328;border:1px solid #d0d7de;font-weight:600;margin-left:6px}
button.secondary:hover{background:#e9edf1}
button:disabled{opacity:.5;cursor:default}
.bar{display:flex;align-items:center;gap:10px;margin:0 0 10px}
.bar .grow{flex:1}
.spinner{width:16px;height:16px;border:2px solid #d0d7de;border-top-color:#1f883d;border-radius:50%;display:inline-block;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}</style></head><body><div class='wrap'>
<h1>selfaudit</h1><p class='sub'>Know whether you can trust your data. Drop a file, paste a URL, or pick a live source — get a clear verdict with the reasons.</p>
<div class='card'>
  <div class='drop' id='drop'><b>Drop a file here</b> — CSV, JSON, or Excel — or click to choose
    <input type='file' id='file' accept='.csv,.tsv,.txt,.json,.xlsx,.xlsm' style='display:none'></div>
  <div class='row'>
    <input type='text' id='url' placeholder='https://host/data.csv'>
    <select id='source'>
      <option value=''>— or a live source —</option>
      <option value='open-meteo'>open-meteo (weather)</option>
      <option value='usgs'>usgs (earthquakes)</option>
      <option value='crypto'>crypto (bitcoin)</option>
    </select>
    <button id='scanBtn' onclick='scan()'>Scan</button>
  </div>
  <div class='examples'>Try an example:
    <button class='secondary' onclick="runExample('weather')">🌤 Weather</button>
    <button class='secondary' onclick="runExample('quakes')">🌍 Earthquakes</button>
    <button class='secondary' onclick="runExample('btc')">₿ Bitcoin</button>
    <button class='secondary' onclick="runExample('sample')">📄 Sample CSV</button>
  </div>
  <p class='muted'>Runs locally — your data never leaves this machine.</p>
</div>
<div class='bar'>
  <span id='spin' class='spinner' style='display:none'></span>
  <span id='status' class='muted'></span>
  <span class='grow'></span>
  <button id='dl' class='secondary' onclick='downloadReport()' disabled>⬇ Download report</button>
</div>
<iframe id='out' title='report'></iframe>
<script>
/*SAMPLE_CSV*/
const out=document.getElementById('out'),drop=document.getElementById('drop'),
file=document.getElementById('file'),urlIn=document.getElementById('url'),
srcIn=document.getElementById('source'),scanBtn=document.getElementById('scanBtn'),
spin=document.getElementById('spin'),statusEl=document.getElementById('status'),
dl=document.getElementById('dl');
let lastReport='';
function show(h){out.srcdoc=h;}
// Keep exactly one input active: touching one clears the others, so switching
// from a file to a live source actually scans the source (not the stale file).
function keepOnly(which){
  if(which!=='file') file.value='';
  if(which!=='url') urlIn.value='';
  if(which!=='source') srcIn.value='';
}
function busy(on,label){spin.style.display=on?'inline-block':'none';scanBtn.disabled=on;
statusEl.textContent=on?(label||'scanning…'):'';}
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over');};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])
{file.files=e.dataTransfer.files;keepOnly('file');scan();}};
file.onchange=()=>{keepOnly('file');scan();};
srcIn.onchange=()=>{keepOnly('source');scan();};
urlIn.oninput=()=>keepOnly('url');
async function post(d){const r=await fetch('/scan',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});return await r.text();}
async function render(label,promise){
  busy(true,label);dl.disabled=true;
  try{const html=await promise;lastReport=html;show(html);dl.disabled=false;}
  catch(e){show('<p style="font:15px system-ui;padding:20px;color:#cf222e">request failed: '+e+'</p>');}
  finally{busy(false);}
}
function readBase64(f){return new Promise((res,rej)=>{const r=new FileReader();
r.onload=()=>res(r.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f);});}
async function scanFile(f){
  const n=f.name.toLowerCase();
  if(n.endsWith('.xlsx')||n.endsWith('.xlsm'))
    return render('scanning '+f.name+'…',post({mode:'xlsx',value:await readBase64(f),name:f.name}));
  return render('scanning '+f.name+'…',post({mode:'csv',value:await f.text(),name:f.name}));
}
async function scan(){
  const f=file.files[0],url=urlIn.value.trim(),src=srcIn.value;
  if(f) scanFile(f);
  else if(url) render('fetching '+url+'…',post({mode:'url',value:url}));
  else if(src) render('fetching '+src+'…',post({mode:'source',value:src}));
  else show('<p style="font:15px system-ui;padding:20px">Choose a file, URL, or source — or try an example.</p>');
}
function runExample(kind){
  keepOnly('none');
  if(kind==='weather') render('fetching live weather…',post({mode:'source',value:'open-meteo'}));
  else if(kind==='quakes') render('fetching recent earthquakes…',post({mode:'source',value:'usgs'}));
  else if(kind==='btc') render('fetching bitcoin prices…',post({mode:'source',value:'crypto'}));
  else render('scanning sample…',post({mode:'csv',value:SAMPLE,name:'sample.csv'}));
}
function downloadReport(){
  if(!lastReport) return;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lastReport],{type:'text/html'}));
  a.download='selfaudit-report.html';a.click();URL.revokeObjectURL(a.href);
}
</script></div></body></html>"""


def _page() -> str:
    """The index page with the sample dataset injected as a JS constant."""
    return _INDEX.replace("/*SAMPLE_CSV*/", "const SAMPLE=" + json.dumps(_SAMPLE_CSV) + ";")


def _error_html(message: str) -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<body style='font:15px system-ui;padding:24px;color:#cf222e'>"
        f"<b>Could not scan.</b><br>{escape(message)}</body>"
    )


def scan_payload(data: dict) -> str:
    """Run a scan for one UI request and return the report HTML (or an error page)."""
    mode = data.get("mode")
    value = data.get("value", "")
    try:
        if mode == "csv":
            ds = parse_text(value, data.get("name", "uploaded.csv"))
        elif mode == "xlsx":
            ds = parse_xlsx_bytes(base64.b64decode(value), name=data.get("name", "uploaded.xlsx"))
        elif mode == "url":
            ds = fetch_csv(value)
        elif mode == "source" and value in _SOURCES:
            ds = _SOURCES[value]()
        else:
            return _error_html(f"unknown request: {mode!r}")
        if ds.n == 0:
            return _error_html("the dataset has no rows")
        report = SelfAuditingDatasetScanner(infer_checks(ds)).scan(ds)
        return report.log.to_html(chart=svg_chart(ds))
    except SourceUnavailable as exc:
        return _error_html(f"source unavailable — {exc}")
    except Exception as exc:  # noqa: BLE001 - any parse/scan error becomes a friendly page
        return _error_html(f"{type(exc).__name__}: {exc}")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._send(_page())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(raw)
        except ValueError:
            self._send(_error_html("invalid request body"), 400)
            return
        self._send(scan_payload(data))

    def log_message(self, *args: object) -> None:  # keep the console quiet
        pass


def build_server(address: tuple[str, int] = ("127.0.0.1", 8000)) -> HTTPServer:
    return HTTPServer(address, _Handler)


def main() -> None:  # pragma: no cover - blocking server loop
    server = build_server()
    host, port = str(server.server_address[0]), server.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"selfaudit UI running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless environments have no browser
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
