你感觉是对的：我上一版是**对准 PDF + 你们 AAS 项目要求**一起写的，但确实偏重了 Agent。按 **PDF 第 28 页 Presentation Guidelines** 来看，中期 pre 其实更像一个 **Software Engineering 项目中期答辩**，Agent 只是其中一部分，不能喧宾夺主。

更准确的结构应该按 PDF 这 19 项来组织：

1. **Project overview and introduction**
   - Kant 是什么、解决什么问题、当前 MVP 状态。

2. **Overall scope using use case diagram**
   - 用户、管理员、外部服务的 use cases。
   - 例如：上传书籍、阅读 EPUB、AI 问答、查看引用、管理对话、查看审计日志。

3. **Project roadmap & key milestones**
   - Phase 0 已完成：需求、架构、backlog。
   - Phase 1 已完成：MVP、核心后端、前端、测试、CI。
   - Phase 2 计划：部署、监控、性能、安全增强、报告完善。

4. **Complete project backlog**
   - 不要只讲 Agent，要列完整 backlog：
     - Frontend reader
     - Book upload
     - Authentication
     - User isolation
     - Conversation management
     - RAG retrieval
     - Agent workflow
     - Storage migration
     - Admin audit
     - CI/CD
     - Testing
     - Security hardening

5. **Overall sprint effort to date**
   - 每个 sprint 做了什么。
   - 每个成员 estimate vs actual effort。
   - 这是 PDF 明确要求，必须有。

6. **Tech stack**
   - Vue 3 / Vite
   - FastAPI
   - PostgreSQL
   - ChromaDB
   - Neo4j
   - OSS / local storage
   - OpenAI-compatible LLM
   - LangGraph / LangChain
   - GitHub Actions
   - Pytest

7. **Architectural constraints and decisions**
   - 用户数据隔离
   - EPUB 大文件处理
   - LLM 幻觉风险
   - 外部 API 依赖
   - 成本与延迟
   - 为什么选 FastAPI、Vue、PostgreSQL、ChromaDB、Neo4j

8. **Logical & physical architecture**
   - Logical：Frontend、API、Domain services、RAG、Agents、Storage、Security。
   - Physical：Browser、Backend server、PostgreSQL、ChromaDB、Neo4j、OSS、LLM API、Notion API、GitHub Actions。

9. **Deployment diagram**
   - 本地 / 云端部署拓扑。
   - 哪些组件容器化，哪些是外部服务。

10. **Microservice architecture / DDD**
   - 如果你们不是微服务，就诚实说是 modular monolith。
   - 用 DDD 解释模块边界：
     - Identity / Auth
     - Library Management
     - Reading Session
     - Conversation
     - RAG / Knowledge Retrieval
     - AI Orchestration
     - Admin / Audit

11. **Software design**
   - 重点讲关键 use case 的 sequence diagram：
     - 上传 EPUB
     - 发起读书问答
     - 管理员查看用户审计
   - Agent 可以在“读书问答”里面出现，但不是全部。

12. **DB / NoSQL design**
   - PostgreSQL：
     - users
     - books
     - notes
     - conversations
     - audit_logs
   - ChromaDB：向量 chunks
   - Neo4j：实体和关系图谱
   - OSS：EPUB 和封面文件

13. **CI/CD pipeline**
   - GitHub Actions：
     - lint
     - backend tests
     - AI security tests
     - frontend build
     - artifacts
   - 这里要展示 pipeline diagram 或截图。

14. **Live demo**
   - 建议 demo 流程：
     - 登录
     - 进入书库
     - 打开 EPUB
     - 选中文本提问
     - 展示回答与引用
     - 展示 prompt injection 拦截
     - 展示测试报告 / CI 结果

15. **Testing**
   - PDF 明确要求：
     - Unit Testing
     - Integration Testing
     - End-to-end Testing
     - Stress Testing
   - 你们现在强项是 backend tests 和 security tests。
   - 如果 E2E / stress 还没做，就列为 Phase 2 plan，不要假装完成。

16. **Management concerns**
   - 团队分工
   - 进度风险
   - 外部服务依赖
   - demo 环境稳定性
   - sponsor / requirement clarification

17. **Technical concerns**
   - LLM 输出不稳定
   - RAG 检索质量
   - Neo4j 图谱抽取准确性
   - OSS / local fallback
   - CI Python 版本和本地 `.venv312` 不一致

18. **Security concerns**
   - Prompt injection
   - Secret leakage
   - Unauthorized file / command request
   - Cross-user data leakage
   - Admin access control
   - Hallucination

19. **AOB / Q&A**
   - 准备老师问：
     - 你们的架构证据在哪里？
     - 目前完成到什么程度？
     - 每个人贡献是什么？
     - 测试怎么证明？
     - 中期之后怎么收尾？