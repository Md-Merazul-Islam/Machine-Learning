import torch

# ১. টেন্সর তৈরি করা
tensor_a = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor_b = torch.tensor([4, 5, 6], dtype=torch.float32)

# ২. টেন্সর যোগ করা
tensor_sum = tensor_a + tensor_b
print("Sum:", tensor_sum)

# ৩. টেন্সর গুন করা (এলিমেন্ট-ওয়াইস মাল্টিপ্লিকেশন)
tensor_product = tensor_a * tensor_b
print("Element-wise Product:", tensor_product)

# ৪. ম্যাট্রিক্স মাল্টিপ্লিকেশন
matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
matrix_mult = torch.mm(matrix_a, matrix_b)
print("Matrix Multiplication:", matrix_mult)

# ৫. টেন্সরের রিশেপ
reshaped_tensor = tensor_a.view(3, 1)
print("Reshaped Tensor:", reshaped_tensor)

# ৬. টেন্সরের গড়
tensor_mean = tensor_a.mean()
print("Mean:", tensor_mean)
