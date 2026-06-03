import urllib.request
import re
import json
import ssl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "swc_registry.json"

def fetch_swc_registry():
    """
    Tự động cào dữ liệu từ kho GitHub của SWC-Registry
    từ SWC-100 đến SWC-136 và build thành file JSON.
    """
    base_url = "https://raw.githubusercontent.com/SmartContractSecurity/SWC-registry/master/entries/docs/SWC-{}.md"
    swc_db = []
    
    print("Đang tiến hành tải và trích xuất dữ liệu SWC-Registry...")
    
    # Bypass SSL verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # ID của SWC dao động từ 100 đến 136
    for swc_id in range(100, 137):
        url = base_url.format(swc_id)
        try:
            # Tải nội dung Raw Markdown từ GitHub
            response = urllib.request.urlopen(url, context=ssl_context)
            markdown_content = response.read().decode('utf-8')
            
            # Trích xuất Title (Nằm dưới tag "# Title")
            title_match = re.search(r'# Title\s+(.+?)(?=\n##|$)', markdown_content, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "Unknown"
            
            # Trích xuất Description (Nằm dưới tag "## Description" cho đến tag "##" tiếp theo)
            desc_match = re.search(r'## Description\s+(.+?)(?=\n##|$)', markdown_content, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Trích xuất Remediation (Nằm dưới tag "## Remediation" cho đến tag "##" tiếp theo)
            rem_match = re.search(r'## Remediation\s+(.+?)(?=\n##|$)', markdown_content, re.DOTALL)
            remediation = rem_match.group(1).strip() if rem_match else ""
            
            swc_db.append({
                "id": f"SWC-{swc_id}",
                "title": title,
                "description": description,
                "remediation": remediation
            })
            print(f"[THÀNH CÔNG] Đã lấy dữ liệu SWC-{swc_id}: {title}")
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[BỎ QUA] SWC-{swc_id} không tồn tại trên hệ thống.")
            else:
                print(f"[LỖI] Lỗi HTTP khi tải SWC-{swc_id}: {e}")
        except Exception as e:
            print(f"[LỖI] Không thể xử lý SWC-{swc_id}: {e}")

    # Lưu dữ liệu ra file JSON
    output_file = DEFAULT_OUTPUT
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(swc_db, f, indent=4, ensure_ascii=False)
        
    print(f"\n[HOÀN TẤT] Đã trích xuất {len(swc_db)} lỗ hổng và lưu vào '{output_file}'.")

if __name__ == "__main__":
    fetch_swc_registry()
