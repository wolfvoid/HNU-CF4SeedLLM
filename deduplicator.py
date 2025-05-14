import numpy as np
from datasketch import MinHash, MinHashLSH
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import os

# 配置loguru日志
logger.add(
    "deduplicator.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


class LSHBERTDeduplicator:
    def __init__(self,
                 bert_model_name='bert-base-chinese',
                 simhash_threshold=0.7,
                 cosine_threshold=0.85,
                 num_perm=128):
        """
        初始化去重器
        :param bert_model_name: BERT模型名称
        :param simhash_threshold: MinHash相似度阈值(θ1)
        :param cosine_threshold: BERT余弦相似度阈值(θ2)
        :param num_perm: MinHash的排列数
        """
        # self.bert_model = SentenceTransformer(bert_model_name)
        self.simhash_threshold = simhash_threshold
        self.cosine_threshold = cosine_threshold
        self.num_perm = num_perm

    def _bert_filter(self, texts):
        """BERT初步过滤"""
        embeddings = self.bert_model.encode(texts)
        # 这里简化处理，实际可以根据业务需求实现更复杂的过滤逻辑
        return texts  # 示例: 过滤掉短文本

    def _create_minhash(self, text):
        """创建文本的MinHash"""
        tokens = text.split()
        mh = MinHash(num_perm=self.num_perm)
        for token in tokens:
            mh.update(token.encode('utf8'))
        return mh

    def _cosine_sim(self, text1, text2):
        """计算两个文本的BERT余弦相似度"""
        emb1 = self.bert_model.encode([text1])
        emb2 = self.bert_model.encode([text2])
        return cosine_similarity(emb1, emb2)[0][0]

    def forward(self, texts):
        """
        执行两阶段文本去重
        :param texts: 原始文本列表
        :return: 去重后的文本列表
        """
        # 第一阶段: BERT判别
        filtered_texts = self._bert_filter(texts)
        print(f"BERT过滤后剩余文本数: {len(filtered_texts)}")
        filtered_texts = texts

        # 初始化结果集和候选集
        D = []  # 结果集
        C = []  # 候选重复集

        # 第二阶段: MinHash+LSH粗过滤
        lsh = MinHashLSH(threshold=self.simhash_threshold,
                         num_perm=self.num_perm)

        # 建立索引
        text_dict = {}  # 存储MinHash对应的原始文本
        for i, text in enumerate(filtered_texts):
            mh = self._create_minhash(text)
            lsh.insert(f"text_{i}", mh)
            text_dict[f"text_{i}"] = text

        # 查询相似文本
        for i, text in enumerate(filtered_texts):
            mh = self._create_minhash(text)
            key = f"text_{i}"

            # 查询相似桶
            result = lsh.query(mh)
            is_dup = False

            # 检查是否有相似文本(排除自身)
            for r in result:
                if r != key:  # 不与自己比较
                    candidate_mh = MinHash(num_perm=self.num_perm)
                    for token in text_dict[r].split():  # text_dict存储了原始文本
                        candidate_mh.update(token.encode('utf8'))
                    sim = mh.jaccard(candidate_mh)
                    if sim >= self.simhash_threshold:
                        is_dup = True
                        break

            if is_dup:
                C.append(text)
            else:
                D.append(text)

        print(f"LSH粗过滤后直接保留文本数: {len(D)}, 候选重复文本数: {len(C)}")

        # 第三阶段: BERT精过滤
        final_D = D.copy()

        # for cand_text in C:
        #     is_dup = False

        #     # 与已保留文本比较
        #     for kept_text in final_D:
        #         sim = self._cosine_sim(cand_text, kept_text)
        #         if sim >= self.cosine_threshold:
        #             is_dup = True
        #             break

        #     if not is_dup:
        #         final_D.append(cand_text)

        # print(f"最终去重后文本数: {len(final_D)}")
        return final_D


# 使用示例
if __name__ == "__main__":
    # 示例文本数据
    texts = [
        "这是第一个测试文本",
        "这是第二个测试文本",
        "这是第一个测试文本",  # 重复文本
        "这是第三个测试文本",
        "这个文本内容完全不同",
        "这是第一个测试文本",  # 重复文本
        "短文本",
        "这是第一个测试文本 带有少量变化"
    ]

    # 创建去重器实例
    deduplicator = LSHBERTDeduplicator(
        bert_model_name='paraphrase-multilingual-MiniLM-L12-v2',
        simhash_threshold=0.5,
        cosine_threshold=0.8
    )

    # 执行去重
    deduplicated_texts = deduplicator.forward(texts)

    print("\n去重结果:")
    for i, text in enumerate(deduplicated_texts):
        print(f"{i+1}. {text}")
