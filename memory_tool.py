"""记忆工具

为HelloAgents框架提供记忆能力的工具实现。
可以作为工具添加到任何Agent中，让Agent具备记忆功能。
"""

import math
from typing import Dict, Any, List, Optional
from datetime import datetime

from hello_agents.tools import Tool, ToolParameter
from hello_agents.memory import BaseMemory, MemoryItem, MemoryManager, MemoryConfig, SQLiteDocumentStore


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口"""

    def __init__(
            self,
            config: Optional[MemoryConfig] = None,
            user_id: str = "default_user",
            enable_working: bool =True,
            enable_episodic: bool =True,
            enable_semantic: bool =True,
            enable_perceptual: bool =False
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        # 初始化存储和检索组件
        self.store = MemoryStore(self.config)
        self.retriever = MemoryRetriever(self.store, self.config)

        # 初始化各类记忆
        self.memory_types = {}

        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config, self.store)
        
        if enable_episodic:
            self.memory_types['episodic'] = EpisdoicMemory(self.config, self.store)
        
        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config, self.store)

        if enable_perceptual:
            self.memory_types['perceptual'] = PerceptualMemory(self.config, self.store)


class MemoryTool(Tool):
    """记忆工具 - 为Agent提供记忆功能"""

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None
    ):
        super().__init__(
            name="memory",
            description = "记忆工具 - 可以存储和检索对话历史、知识和经验"
        )

        #初始化记忆管理器
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_config,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types
        )




    


    def execute(self,action: str, **kwargs) ->str:
        """执行记忆操作

        支持的操作：
        - add: 添加记忆(支持4种类型： working/episodic/semantic/perceptual)
        - search: 搜索记忆
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆(多种策略)
        - consolidate: 整合记忆(短期->长期)
        - clear_all:清空所有记忆
        """

        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_summary(**kwargs)
    
    def _add_memory(
            self,
            content: str = "",
            memory_type: str = "working",
            importance: float = 0.5,
            file_path: str = None,
            modality: str = None,
            **metadata
    ) -> str:
        """添加记忆"""
        try:
            # 确保会话DI存在
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 感知记忆文件支持
            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)
            
            # 添加会话信息到元数据
            metadata.update({
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            })

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type = memory_type,
                importance = importance,
                metadata = metadata,
                auto_classify=False
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        
        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"
        
    def _search_memory(
            self,
            query: str,
            limit: int = 5,
            memory_types: List[str] = None,
            memory_type: str = None,
            min_importance: float = 0.1
    ) -> str:
        """搜索记录"""
        try:
            #参数标准化处理
            if memory_type and not memory_types:
                memory_types = [memory_type]
            
            results = self.memory_manager.retrieve_memories(
                query = query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance
            )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"
            
            #格式化结果
            formatted_results = []
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.memory_type, memory.memory_type)

                content_preview = memory.content[:80] + "..:" if len(memory.content) > 80 else memory.conetnt
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"
    
    def _forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> str:
        """遗忘记忆（支持多种策略）"""
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
            )
            return f"🧹 已遗忘 {count} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"

    def _consolidate(self, from_type: str = "working", to_type: str = "episodic", importance_threshold: float = 0.7) -> str:
        """整合记忆（将重要记忆提升为长期记忆）"""
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold
            )
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败：{str(e)}"



#=============== 四种记忆类型 ===============

class WorkingMemory:
    """
    工作记忆实现
    特点：
    - 容量有限(默认50条) + TTL自动清理
    - 纯内存存储，访问速度记忆
    - 混合检索： TF-IDF向量化 + 关键词匹配
    """

    def __init__(self, config:MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl or 60
        self.memories = []
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆"""
        self._expire_old_memories() # 过期清理, 添加记忆前，先扔掉过期的记忆

        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory() # 容量管理：如果当前记忆超过最大容量的话，再扔掉不重要的记忆

        self.memories.append(memory_item) # 最后才是添加记忆
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：TF-IDF向量化 + 关键词匹配"""
        self._expire_old_memories() # 检索前先清理过期数据

        # 尝试TF-IDF向量搜索
        vector_scores = self._try_tfidf_search(query)

        # 计算综合分数
        # 最终得分公式为：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)。
        scored_memories = []
        for memory in self.memories:  #注意：这里搜索的是self.memories 这个内存列表，而不是数据库
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content) #关键词匹配

            #混合评分
            base_relevance = vector_score *0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score #如果vector_score为0就退化为关键词匹配，如果TF-IDF有结果（vector_score > 0）：混合使用，TF-IDF占70%，关键词匹配占30%
            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)

            final_score = base_relevance * time_decay * importance_weight

            if final_score > 0:
                scored_memories.append(final_score, memory)
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

class EpisodicMemory:
    """情景记忆实现
    特点：
    - SQLite+Qdrant混合存储架构，SQLite负责结构化预过滤(筛掉时间不对、会话不对的)，Qdrant负责语义向量检索(在剩下的里面找意思最像的)
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """

    def __init__(self, config: MemoryConfig):
        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)
        self.embedder = create_embedding_model_with_fallback()
        self.sessions = {}  # 会话索引

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 创建情景对象
        episode = Episode(
            episode_id=memory_item.id,
            session_id=memory_item.metadata.get("session_id","default"),
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=memory_item.metadata
        )

        # 更新会话索引
        session_id = episode.session_id
        if session_id not in self.sessions: # self.sessions 是一个放在内存里的字典，相当于一个快速导航表：
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        #持久化存储 （SQLite + Qdrant）
        self._persist_episode(episode)
        return memory_item.id
    
    def retrieve(self, query: str, limit: int =5, **kwargs) -> List[MemoryItem]:
        """混合检索： 结构化过滤 + 语义向量检索"""
        # 1. 结构化预过滤（时间范围、重要性等）
        candidate_ids = self._structured_filter(**kwargs)

        # 2. 向量语义检索
        hits = self._vector_search(query, limit * 5, kwargs.get("user_id"))

        # 3. 综合评分与排序
        results = []
        for hit in hits:
            if self._should_include(hit, candidate_ids, kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item(hit)
                results.append((score, memory_item))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

    def _calculate_episode_score(self, hit) -> float:
        """情景记忆评分算法"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"]["timestamp"])
        importance = hit["metadata"].get("importance",0.5)

        # 评分公式：(向量相似度 * 0.8 + 时间近因性 * 0.2) * 重要性权重
        base_relevance = vec_score * 0.8 + recency_score * 0.2 # 语义匹配度占80%权重，时间近因性占20%权重
        importance_weight = 0.8 + (importance * 0.4)

        return base_relevance * importance_weight


class SemanticMemory(BaseMemory):
    """语义记忆实现

    特点：
    - 使用HuggingFace中文预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 嵌入模型（统一提供）
        self.embedding_model = get_text_embedder()

        # 专业数据库存储
        self.vector_store = QdrantConnectionManager.get_instance(**qdrant_config)
        self.graph_store = Neo4jGraphStore(**neo4j_config)

        # 实体和关系缓存
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # NLP处理器（支持中英文）
        self.nlp = self._init_.nlp()
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        # 1. 生成文本嵌入
        embedding = self.embedding_model.encode(memory_item.content)

        # 2. 提取实体和关系
        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)

        # 3. 存储到Neo4j数据库
        for entity in entities:
            self._add_entity_to_graph(entity, memory_item)
        
        for relation in relations:
            self._add_relation_to_graph(relation, memory_item)
        
        # 4. 存储到Qdrant向量数据库
        metadata = {
            "memory_id": memory_item.id,
            "entities": [e.entity_id for e in entities],
            "entity_count": len(entities),
            "relation_count": len(relations)
        }

        self.vector_store.add_vectors(
            vectors=[embedding.tolist()],
            metadata=[metadata],
            ids=[memory_item.id]
        )

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆"""
        # 1. 向量检索
        vector_results = self._vector_search(query, limit * 2, user_id)

        # 2. 图检索
        graph_results = self._graph_search(query, limit * 2, userid)

        # 3. 混合排序
        combined_results = self._combine_and_rank_results(
            vector_results, graph_results, query, limit
        )

        return combined_results[:limit]
    
    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        """混合排序结果"""
        combined = {}

        #合并向量和图检索结果
        for result in vector_results:
            combined[result["memory_id"]] = {
                **result,
                "vector_score": result.get("score",0.0),
                "graph_score": 0.0
            }
        
        for result in graph_results:
            memory_id = result["memory_id"]
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            else:
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0)
                }
        
        #计算混合分数
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)

            # 基础相似度得分
            base_relevance = vector_score *0.7 + graph_score * 0.3

            #重要性权重 [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)

            #最终得分：相似度 * 重要性权重
            combined_score = base_relevance * importance_weight
            result["combined_score"] = combined_score
        
        # 排序并返回
        sorted_results = sorted(
            combined.values(),
            key = lambda x: x["combined_score"],
            reverse=True
        )

        return sorted_results[:limit]

class PerceptualMemory(BaseMemory):
    """感知记忆实现

    特点：
    - 支持多模态数据
    - 跨模态相似性搜索
    - 支持内容生成和检索
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        #多模态编码器
        self.text_embedder = get_text_embedder()
        self._clip_model = self._init_clip_model() # 图像编码
        self._clap_model = self._init_clap_model() #音频编码

        # 按模态分高的向量存储
        self.vector_stores = {
            "text": QdrantConnectionManager.get_instance(
                collection_name="perceptual_text",
                vector_size=self.vector_dim
            ),
            "image": QdrantConnectionManager.get_instance(
                collection_name="perceptual_image",
                vector_size=self._image_dim
            ),
            "auto": QdrantConnectionManager.get_instance(
                collection_name="perceptual_audio",
                vector_size=self._audio_dim
            )
        }
    
    def retrieve(self, query: sr, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆(可筛模态：同模态向量检索+时间/重要性融合)"""
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality = kwargs.get("query_modality", target_modality or "text")

        # 同模态向量检索
        try:
            query_vector = self._encode_data(query, query_modality)
            store = self._get_vector_store_for_modality(target_modality or query_modality)
            
            where = {"memory_type":"perceptual"}
            if user_id:
                where["user_id"] = user_id
            if target_modality:
                where["modality"] = target_modality
            
            hits = store.search_similar(
                query_vector=query_vector,
                limit=max(limit * 5, 20),
                where=where
            )
        
        except Exception:
            hits = []
        
        # 融合排序(向量相似度 + 时间近因性 + 重要性权重)
        results = []
        for hit in hits:
            vector_score = float(hit.get("score", 0.0))
            recency_score = self._calculate_recenecy_score(hit["metadata"]["timestamp"])
            importance = hit["metadata"].get("importance", 0.5)

            # 评分算法
            base_relevance = vector_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight

            results.append((combined_score, self._create_memory_item(hit)))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]
    
    def _calculate_recency_score(self, timestamp: str) -> float:
        """计算时间近因性得分"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600

            # 指数衰减：24小时内保持高分，之后逐渐衰减
            decay_factor = 0.1 #衰减系数
            recency_score = math.exp(-decay_factor * age_hours / 24)
        
        except Exception:
            return 0.5 #默认中等分数