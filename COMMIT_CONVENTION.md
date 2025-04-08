# Quy tắc viết Commit Message

## Cấu trúc cơ bản

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Các quy tắc chính

1. **Type**: Loại commit (bắt buộc)

    - `feat`: Thêm tính năng mới
    - `fix`: Sửa lỗi
    - `docs`: Thay đổi tài liệu
    - `style`: Thay đổi format code (không ảnh hưởng đến logic)
    - `refactor`: Tái cấu trúc code
    - `test`: Thêm hoặc sửa test
    - `chore`: Các thay đổi nhỏ khác

2. **Scope**: Phạm vi ảnh hưởng (tùy chọn)

    - `model`: Các thay đổi liên quan đến model
    - `data`: Xử lý dữ liệu
    - `train`: Logic huấn luyện
    - `utils`: Công cụ hỗ trợ
    - `config`: Cấu hình
    - `deps`: Dependencies

3. **Subject**: Mô tả ngắn gọn (bắt buộc)

    - Không quá 50 ký tự
    - Viết ở thì hiện tại
    - Không viết hoa chữ đầu
    - Không có dấu chấm ở cuối

4. **Body**: Mô tả chi tiết (tùy chọn)

    - Giải thích lý do WHAT và WHY (không cần HOW)
    - Mỗi dòng không quá 72 ký tự
    - Cách 1 dòng với subject

5. **Footer**: Thông tin bổ sung (tùy chọn)
    - Các breaking changes
    - Các issues được đóng
    - Người review

## Ví dụ

```
feat(model): thêm layer attention vào mô hình

- Thêm multi-head attention layer sau LSTM
- Cải thiện độ chính xác từ 85% lên 87%
- Tối ưu hóa memory usage

Closes #123
Reviewed-by: @mentor
```

```
fix(data): sửa lỗi xử lý dữ liệu thiếu

Thêm xử lý cho trường hợp giá trị null trong dataset
để tránh crash khi training

Fixes #456
```

```
docs(readme): cập nhật hướng dẫn cài đặt
```

## Lưu ý quan trọng

1. **Commit nhỏ và tập trung**

    - Mỗi commit chỉ nên giải quyết một vấn đề
    - Tránh gộp nhiều thay đổi không liên quan

2. **Viết message rõ ràng**

    - Người khác đọc có thể hiểu được bạn đã làm gì
    - Giải thích lý do của thay đổi nếu cần thiết

3. **Sử dụng tiếng Việt**

    - Viết không dấu nếu gặp vấn đề về encoding
    - Giữ nhất quán trong toàn bộ dự án

4. **Tham chiếu issues**

    - Sử dụng từ khóa như "Fixes", "Closes", "Resolves"
    - Thêm số issue sau # (ví dụ: #123)

5. **Review code trước khi commit**
    - Kiểm tra lại các thay đổi
    - Đảm bảo không có debug code
    - Xóa các comment không cần thiết
