# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng cờ `assistant_only_loss=True` trong thư viện TRL hóa ra lại âm thầm giám sát **0 token** (hoặc crash) trên các dòng mô hình mới như Qwen3.5 do template thiếu thẻ `{% generation %}`. Trước đây tôi luôn đinh ninh rằng việc truyền một cờ tích hợp sẵn của Hugging Face là đủ an toàn, nhưng thực tế nếu không giải mã ngược nhãn token (`decode_supervised`), mô hình có thể đã trải qua hàng giờ huấn luyện trên con số không mà không hề có bất kỳ cảnh báo lỗi nghiêm trọng nào.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu kiểm chứng tính nhất quán giữa prompt huấn luyện và prompt đánh giá (Finding F-31). Ban đầu tôi dự đoán khâu huấn luyện LoRA và cân chỉnh VRAM sẽ tốn nhiều thời gian nhất. Nhưng thực tế, việc định dạng lệch giữa prompt huấn luyện (vốn nhồi nhét toàn bộ schema vào lượt user) và prompt đánh giá (dùng system prompt rút gọn) đã khiến mô hình fine-tune ban đầu sinh văn xuôi và đạt điểm 0 tròn trĩnh, dù hàm loss hội tụ rất đẹp. Việc gỡ lỗi này đòi hỏi phải so sánh chuỗi giải mã từng token chứ không thể nhìn biểu đồ loss.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng: "Nếu mô hình fine-tune học kém, cứ tăng rank LoRA lên (từ r=8 lên r=64, r=128) thì chắc chắn sẽ cải thiện". Thí nghiệm đối chứng `attn_only` (r=283) so với `correct` (r=16) trên cùng một lượng tham số đã đập tan hoàn toàn niềm tin đó: rank cao dồn vào lớp attention chỉ làm mô hình học vẹt và ép loss ảo, trong khi việc dàn trải LoRA ra tất cả các module linear (`text-linear`) mới là chìa khóa đem lại độ chính xác tổng quát hóa cao.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ phân tích code xử lý token offset trong `data.py`, tính toán ngân sách tham số LoRA và kiểm tra cú pháp của chat template. Chỗ AI assistant hay đưa ra gợi ý sai nhất là thường xuyên tự động thêm cờ `bf16=True` hoặc đề xuất dùng `bitsandbytes` 4-bit theo thói quen mặc định trên phần cứng hiện đại (A100), trong khi môi trường thực tế (Tesla T4 hoặc macOS) yêu cầu gradient scaling fp16 và không hỗ trợ các kernel này.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở máy ảo hay tải trọng số mô hình, mà là **đóng băng một tập dữ liệu kiểm thử độc lập (golden eval set) và đo đạc baseline từ một prompt được tối ưu hóa kỹ lưỡng (Prompt Engineering)**. Nếu một system prompt tốt đã đạt 80–90% yêu cầu nghiệp vụ với chi phí vận hành và bảo trì bằng 0, tôi sẽ khuyên khách hàng không nên vội vã fine-tune. Nếu bắt buộc fine-tune, baseline đó sẽ là chiếc mỏ neo trung thực duy nhất để chứng minh giá trị của dự án.
