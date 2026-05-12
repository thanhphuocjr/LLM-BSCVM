import json
import os

def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None

def extract_data(raw_data, dataset_name):
    """Trích xuất dữ liệu dựa trên cấu trúc thực tế của Dappscan và Solodit."""
    processed_corpus = []
    
    # Kiểm tra xem dữ liệu có phải là list không
    if isinstance(raw_data, list):
        for item in raw_data:
            code = ""
            label = "Safe"
            
            if dataset_name == "Solodit":
                # Theo sample: key chứa code là "code", key chứa nhãn là "label"
                code = item.get("code", "")
                
                if "label" in item:
                    # Chuẩn hóa nhãn (viết hoa chữ cái đầu cho đồng bộ)
                    raw_label = str(item.get("label")).strip().capitalize()
                    label = "Safe" if raw_label.lower() == "safe" else "Vulnerable"
                    
            elif dataset_name == "Dappscan":
                # Theo sample: key chứa code là "Code", key chứa nhãn là "Output"
                code = item.get("Code", "")
                
                if "Output" in item:
                    # Lấy nhãn trực tiếp từ trường "Output"
                    label = str(item.get("Output")).strip()
            
            # Chỉ thêm vào danh sách nếu trích xuất được code hợp lệ
            if code and isinstance(code, str) and len(code.strip()) > 0:
                processed_corpus.append({
                    "Code": code,
                    "Label": label
                })
                
    else:
        print(f"Cảnh báo: Cấu trúc file {dataset_name} không phải là một List các object. Bạn cần kiểm tra lại định dạng file.")
        
    return processed_corpus

def main():
    # 1. Đường dẫn file (Bạn giữ nguyên đường dẫn trên máy bạn)
    solodit_path = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/Solodit/solodit.json"
    dappscan_path = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/Dappscan/dappscan.json"
    output_path = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/smart_contract_corpus.json"

    print("Đang tiến hành đọc dữ liệu từ ổ cứng...")
    solodit_raw = load_json(solodit_path)
    dappscan_raw = load_json(dappscan_path)

    corpus = []

    # 2. Xử lý tập Solodit
    if solodit_raw:
        print("\nĐang xử lý tập Solodit...")
        solodit_processed = extract_data(solodit_raw, "Solodit")
        corpus.extend(solodit_processed)
        print(f"  -> Trích xuất thành công {len(solodit_processed)} hợp đồng từ Solodit.")

    # 3. Xử lý tập Dappscan
    if dappscan_raw:
        print("\nĐang xử lý tập Dappscan...")
        dappscan_processed = extract_data(dappscan_raw, "Dappscan")
        corpus.extend(dappscan_processed)
        print(f"  -> Trích xuất thành công {len(dappscan_processed)} hợp đồng từ Dappscan.")

    # 4. Ghi file kết quả gộp
    if corpus:
        print(f"\nĐang tổng hợp và lưu kho dữ liệu ({len(corpus)} hợp đồng) vào: {output_path}")
        try:
            # Tạo folder nếu chưa tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ghi ra JSON với format đẹp, giữ nguyên font chữ chuẩn Unicode
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, ensure_ascii=False, indent=4)
            print("🎉 THÀNH CÔNG! Đã tạo xong Smart Contract Corpus đúng chuẩn của bài báo.")
        except Exception as e:
            print(f"Lỗi khi ghi file: {e}")
    else:
        print("\n❌ Không có hợp đồng nào được trích xuất, hãy kiểm tra lại!")

if __name__ == "__main__":
    main()