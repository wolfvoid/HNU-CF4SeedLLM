import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Set, Optional
from PIL import Image
import io
import torch
from nltk.tokenize import sent_tokenize
import json
import numpy
from loguru import logger
import torchvision.transforms as transforms
import re
import os
import shutil

# 配置loguru日志
logger.add(
    "CleanFrame.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def convert_cid_to_text(text):
    return re.sub(r'$cid:(\d+)$', lambda m: chr(int(m.group(1))), text)


class PDFMultimodalProcessor:
    def __init__(self,):
        """
        初始化PDF多模态处理器
        :param text_chunk_size: 文本分块长度（字符数）
        :param device: 计算设备
        """

        self.text_chunk_size = 300
        self.overlap_ratio = 0.1
        assert self.overlap_ratio < 1.0, "重叠比例必须小于1"
        assert self.text_chunk_size > 50, "分块尺寸过小会导致语义丢失"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.templates = []  # 提示模板集

        self.current_pdf_path = None  # 当前处理的PDF路径 当前处理，dynamic

        # 新增表格检测相关初始化
        self.detection_transform = transforms.Compose([
            self.MaxResize(max_size=800),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 初始化模型
        logger.info("初始化加载模型文件，可能需要一点时间")
        cache_dir = "./model_cache"  # 指定下载路径
        # 加载BLIP模型
        blip_model_name = "IDEA-CCNL/Taiyi-BLIP-750M-Chinese"
        self.blip_model, self.blip_processor = self._load_blip_model(
            blip_model_name, cache_dir)
        # 加载表格检测模型
        table_model_name = "microsoft/table-transformer-detection"
        self.table_model = self._load_table_model(table_model_name, cache_dir)

        logger.info(f"当前设备: {self.device}")
        logger.info("启用：使用BLIP模型进行图像处理")
        logger.info("启用：使用table模型进行表格处理")

    # 加载模型

    def _load_blip_model(self, model_name: str = None, cache_dir: str = None) -> Tuple[torch.nn.Module, callable]:
        """加载BLIP模型和处理器"""
        try:
            from transformers import AutoProcessor, BlipForConditionalGeneration
            processor = AutoProcessor.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=cache_dir  # 指定下载路径
            )
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                cache_dir=cache_dir  # 指定下载路径
            ).to(self.device)
            logger.info("BLIP模型加载成功")
            return model.eval(), processor
        except Exception as e:
            logger.error(f"BLIP模型加载失败: {str(e)}")
            raise RuntimeError(
                f"BLIP加载失败: {str(e)}\n请安装: pip install transformers")

    def _load_table_model(self, model_name: str = None, cache_dir: str = None) -> torch.nn.Module:
        """加载表格检测模型"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 临时忽略所有警告
            try:
                from transformers import AutoModelForObjectDetection
                model = AutoModelForObjectDetection.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,  # 指定下载路径
                    local_files_only=True,
                )
                model.load_state_dict(model.state_dict(), assign=True)
                model.to(self.device)
                logger.info("Table Transformer模型加载成功")
                return model.eval()
            except Exception as e:
                logger.error(f"表格模型加载失败: {str(e)}")
                raise RuntimeError(
                    f"model-transformer加载失败: {str(e)}\n请安装: pip install transformers")

    # 分文件处理
    def process_folder(self, input_folder: str) -> Dict[str, dict]:
        """
        处理输入文件夹中的所有PDF文件
        :param input_folder: 输入文件夹路径
        :return: 字典{文件名: 处理结果}
        """
        import os
        results = {}
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.pdf'):
                logger.info(f">>处理文件: {filename}")
                pdf_path = os.path.join(input_folder, filename)
                results[filename] = self.process_single_pdf(pdf_path)
        return results

    def process_single_pdf(self, pdf_path: str) -> dict:
        """
        处理单个PDF文件
        :param pdf_path: PDF文件路径
        :return: 多模态内容集合字典
        """
        # 初始化结果集合
        result = {
            'text_chunks': [],  # 对应S_t
            'images': [],       # 对应S_g
            'tables': []        # 对应S_f
        }

        # 提取所有内容
        images, tables, text = self._extract_all_content(pdf_path)

        # 处理文本
        logger.info(">>>>处理文本内容")
        result['text_chunks'] = self._process_text(text)

        # 处理图像
        logger.info(">>>>处理图像内容")
        for img_data in images:
            processed_img = self._process_image(img_data)
            if processed_img:
                result['images'].append(processed_img)

        # 处理表格
        logger.info(">>>>处理表格内容")
        for table_data in tables:
            processed_table = self._process_table(table_data)
            if processed_table:
                result['tables'].append(processed_table)

        return result

    # 提取PDF中的所有内容（图像、表格、文本）
    def _extract_all_content(self, pdf_path: str) -> Tuple[List[dict], List[dict], str]:
        """
        提取PDF中的所有内容（图像、表格、文本）
        :return: (图像列表, 表格列表, 全文文本)
        """

        doc = fitz.open(pdf_path)
        images = []
        tables = []
        full_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 提取文本
            temp_text = page.get_text("text").replace(
                "\n", " ").replace("\r", " ").replace(" ", "").strip()
            full_text += temp_text

            # 提取图像
            images.extend(self._extract_images_from_page(page, doc, page_num))

        # 提取表格
        tables.extend(self._extract_tables(pdf_path))

        doc.close()
        return images, tables, full_text

    # 处理文本、图像和表格

    def _process_text(self, paragraphs: str) -> List[dict]:
        """按指定的 chunk_size 和 overlap_size 交叠切割文本"""
        # chunk_size 和 over_size需要自己尝试
        # 一般设置 overlap_size = chunk_size * 10%-20%
        chunk_size = self.text_chunk_size
        overlap_size = int(chunk_size * self.overlap_ratio)
        sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            overlap = ''
            prev_len = 0
            prev = i-1
            # 向前计算重叠部分
            while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
                overlap = sentences[prev] + ' ' + overlap
                prev -= 1
            chunk = overlap + chunk
            next = i+1
            # 向后计算重叠部分
            while next < len(sentences) and len(chunk)+len(sentences[next]) <= chunk_size:
                chunk += ' ' + sentences[next]
                next += 1
            chunk = chunk.replace('\n', ' ').replace(
                '\r', ' ').replace(' ', '').strip()
            chunks.append(chunk)
            i = next
        return chunks

    def _process_image(self, img_data: dict) -> Optional[dict]:
        """使用BLIP生成图像描述"""
        try:
            # 确保图像为RGB模式
            img = img_data["image"].convert('RGB')
            if max(img.size) > 1024:
                img = img.resize((800, 800))  # 调整过大图像尺寸

            # BLIP处理流程
            inputs = self.blip_processor(
                images=img,
                text="这是一张关于",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(
                    # **inputs,
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,  # 控制描述长度
                    num_beams=1,     # 提高生成质量
                )

            description = self.blip_processor.decode(
                out[0],
                skip_special_tokens=True
            )

            description = description.replace(
                "这是一张关于", "").replace("的图像", "").replace(" ", "").strip()

            return {
                "page": img_data["page"],
                "index": img_data["index"],
                "description": description,
                "metadata": img_data["metadata"]
            }
        except Exception as e:
            print(f"图像描述生成失败（页码{img_data.get('page', '未知')}）：{str(e)}")
            return None

    def _process_table(self, table_data: dict) -> Optional[dict]:
        """处理表格（实现算法中的双重表示）"""
        try:
            # 表格转图像
            table_image = self._crop_table_image(
                table_data["page"],
                table_data["bbox"]
            )
            table_image = table_image.convert('RGB')
            if max(table_image.size) > 1024:
                table_image = table_image.resize((800, 800))

            # 复用图像处理流程
            img_desc = self._process_image({
                "page": table_data["page"],
                "image": table_image,
                "index": f"table_{table_data.get('index', 0)}",
                "metadata": table_data["metadata"]
            })

            # 组合最终结果
            return {
                "page": table_data["page"],
                "description": img_desc["description"] if img_desc else "",
                # "markdown": table_data["markdown"],
                # "raw_data": table_data["raw_data"],
                "table_image": table_image
            }
        except Exception as e:
            logger.error(f"表格处理失败（页码{table_data.get('page', '未知')}）：{str(e)}")
            return None

    # 处理图像辅助函数
    def _extract_images_from_page(self, page, doc, page_num: int) -> List[dict]:
        """从页面提取图像"""
        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list, 1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # 使用Pillow保存图像
            image = Image.open(io.BytesIO(image_bytes))

            images.append({
                "page": page_num + 1,
                "index": img_index,
                "image": image,
                "metadata": base_image
            })

        return images

    # 处理表格辅助函数
    # 新增图像预处理转换类
    class MaxResize:
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            return image.resize((int(round(width*scale)), int(round(height*scale))))

    # 坐标转换静态方法
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        boxes = PDFMultimodalProcessor.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * \
            torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return boxes

    # 模型输出转换方法
    def outputs_to_objects(self, outputs, img_size):
        id2label = self.table_model.config.id2label
        id2label[len(id2label)] = "no object"
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [elem.tolist()
                       for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        return [{
            "label": id2label[int(label)],
            "score": float(score),
            "bbox": [float(e) for e in bbox]
        } for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes)
            if id2label[int(label)] != "no object"]

    def _crop_table_image(self, page_num: int, bbox: Tuple[float]) -> Image.Image:
        """根据检测到的表格坐标裁剪图像"""
        doc = fitz.open(self.current_pdf_path)  # 需在类中维护当前处理路径
        page = doc[page_num - 1]

        # 高分辨率渲染
        matrix = fitz.Matrix(2, 2)  # 缩放系数提升检测精度
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # 坐标转换 (PDF坐标到像素坐标)
        x0, y0, x1, y1 = [int(coord * 2) for coord in bbox]  # 注意矩阵缩放后的坐标映射
        cropped_img = img.crop((x0, y0, x1, y1))

        doc.close()
        return cropped_img

    def _extract_tables(self, pdf_path: str) -> List[dict]:
        """使用Table Transformer检测表格"""
        self.current_pdf_path = pdf_path  # 维护当前处理路径
        if not self.table_model:
            return []

        doc = fitz.open(pdf_path)
        tables = []

        for page_num in range(len(doc)):
            # 将PDF页面渲染为图像
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 提高分辨率
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # 执行表格检测
            try:
                pixel_values = self.detection_transform(
                    img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.table_model(pixel_values)

                # 解析检测结果
                detected_tables = self.outputs_to_objects(outputs, img.size)

                # 转换数据结构
                for idx, table in enumerate(detected_tables):
                    table_image = img.crop(table["bbox"])
                    # 输出图片
                    # table_image.show(
                    #     title=f"Page {page_num + 1} - Table {idx + 1}")
                    tables.append({
                        "page": page_num + 1,
                        "index": idx,
                        "bbox": table["bbox"],
                        "image": table_image,
                        "score": table["score"],
                        "metadata": {
                            "detection_score": table["score"],
                            "model": "table-transformer"
                        }
                    })

            except Exception as e:
                logger.error(f"第{page_num+1}页表格检测失败: {str(e)}")

        doc.close()
        return tables

    def forward(self, input_folder: str) -> None:
        """整合处理流程并生成文本集合JSON文件
        :param input_folder: 输入文件夹路径
        :param output_json: 输出JSON文件路径（默认当前目录output.json）
        """
        # 处理所有PDF文件
        result_path = "./output_text"
        if os.path.exists(result_path):
            shutil.rmtree(result_path)  # 删除整个文件夹
            os.makedirs(result_path)   # 重新创建空文件夹
            print(f"已清空文件夹: {result_path}")
        else:
            print(f"文件夹不存在: {result_path}")
        results = self.process_folder(input_folder)

        # 创建三个独立的列表来存储不同类型的内容
        all_texts = []
        all_images = []
        all_tables = []
        combined_texts = []

        # 聚合所有内容
        for filename, data in results.items():
            all_texts.extend(data['text_chunks'])
            all_images.extend([img['description'] for img in data['images']])
            all_tables.extend([table['description']
                              for table in data['tables']])

            # 整合所有文本
            combined_texts.extend(data['text_chunks'])
            combined_texts.extend([img['description']
                                  for img in data['images']])
            combined_texts.extend([table['description']
                                   for table in data['tables']])

        # 保存到各自的JSON文件
        def save_to_json(data, filename):
            with open(f"./result/{filename}", 'w', encoding='utf-8') as f:
                for item in data:
                    json_line = json.dumps({"text": item}, ensure_ascii=False)
                    f.write(json_line + '\n')

        save_to_json(all_texts, "texts.json")
        save_to_json(all_images, "images.json")
        save_to_json(all_tables, "tables.json")

        # 保存整合后的统一文本集合
        with open("./result/result.json", 'w', encoding='utf-8') as f:
            for text in combined_texts:
                json_line = json.dumps({"text": text}, ensure_ascii=False)
                f.write(json_line + '\n')

        for filename, result in results.items():
            logger.info(f"Processed {filename}:")
            logger.info(f"Text chunks: {len(result['text_chunks'])}")
            logger.info(f"Images: {len(result['images'])}")
            logger.info(f"Tables: {len(result['tables'])}")

        logger.info(f"文本聚合完成:")
        logger.info(f"- 文本内容: {len(all_texts)}条 -> ./result/texts.json")
        logger.info(f"- 图像描述: {len(all_images)}条 -> ./result/images.json")
        logger.info(f"- 表格描述: {len(all_tables)}条 -> ./result/tables.json")
        logger.info(f"整合结果 -> ./result/result.json")


if __name__ == "__main__":
    # 示例用法
    processor = PDFMultimodalProcessor()
    results = processor.forward(
        r"D:\HNU-CF4SeedLLM\test")
