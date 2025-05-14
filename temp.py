import fitz
doc = fitz.open(r"D:\HNU-CF4SeedLLM\test\播期对寒地粳稻产量及温光资源利用的影响.pdf")
doc.subset_fonts()  # 嵌入所有字体子集
doc.save(r"D:\HNU-CF4SeedLLM\test\播期对寒地粳稻产量及温光资源利用的影响22.pdf",
         garbage=4, deflate=True)
