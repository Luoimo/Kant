# Individual Project Report: DeepReadAgent

**Project Title:** Kant: Agentic AI Deep Reading and Knowledge Management System  
**Team Number:** 22  
**Individual Name:** [Your Name]  
**Agent Responsible For:** DeepReadAgent

## 1. Introduction

Kant is an agentic AI deep reading and knowledge management system designed to help users read EPUB books more critically and productively. Instead of providing a single chatbot function, Kant uses a coordinated multi-agent architecture to support book ingestion, evidence-based question answering, citation generation, answer review, note generation, follow-up question generation, and security monitoring.

My individual responsibility in this project was the **DeepReadAgent**, which acts as the main question-answering agent in the system. Its purpose is to receive a user question about a book, retrieve relevant textual evidence from the uploaded book collection, and generate a grounded answer with citations. In the overall workflow, DeepReadAgent is the core agent that transforms the user's reading question into an explainable response supported by retrieved book content.

The main objectives of DeepReadAgent are:

- To provide deep reading answers based on the content of the current book.
- To reduce hallucination by grounding responses in retrieved textual evidence.
- To support multi-turn reading conversations through short-term conversation state.
- To use relevant user context, such as selected text, current chapter, and memory context.
- To return citations so that users can verify the source of the answer.
- To support streaming responses for a smoother reading experience.

Through this agent, my work contributed directly to the project's requirements in agentic AI system design, explainable and responsible AI, AI security, and LLMSecOps-oriented engineering practice.

## 2. Personal Contribution Summary

The DeepReadAgent is a central part of the Kant system because it is responsible for the main evidence-grounded reading response. My contribution focused on designing and implementing this agent so that it could work as a controlled, explainable, and security-aware component rather than a simple LLM call.

| Contribution Area | My Work | Related Files | Outcome |
|---|---|---|---|
| Agent workflow design | Designed the DeepReadAgent as a tool-using ReAct agent for book-specific question answering. | `backend/agents/deepread_agent.py` | The system can answer reading questions through a structured retrieval and generation workflow. |
| Retrieval integration | Integrated book content search through hybrid retrieval and scoped tool access. | `backend/agents/deepread_agent.py`, `backend/rag/retriever/hybrid_retriever.py` | Answers are grounded in retrieved evidence instead of relying only on model memory. |
| Dynamic reading context | Added support for current book, current chapter, selected text, and memory context. | `backend/agents/deepread_agent.py` | The agent can respond in the context of the user's active reading session. |
| Citation support | Connected retrieved documents with citation construction. | `backend/agents/deepread_agent.py`, `backend/xai/citation.py` | Users can verify generated answers against source evidence. |
| Streaming response support | Implemented event streaming for token output, tool status, and completion metadata. | `backend/agents/deepread_agent.py`, `backend/api/chat.py` | The frontend can show a smoother real-time reading assistant experience. |
| Security-aware boundary | Limited DeepReadAgent tools to retrieval-oriented actions. | `backend/agents/deepread_agent.py`, `backend/security/input_filter.py` | The agent does not receive arbitrary file, command, or system execution capabilities. |

## 3. Agent Design

### 3.1 Agent Role and Functionality

DeepReadAgent is the primary answering agent in Kant. It is responsible for handling book-related questions after the request has passed the safety layer and, when applicable, the routing layer. The agent does not answer only from the model's internal knowledge. Instead, it uses a retrieval-augmented generation workflow to search the user's uploaded book content and then generate an answer based on the retrieved evidence.

The agent supports several reading scenarios:

- Asking questions about the current book.
- Explaining a selected passage.
- Analysing a chapter or concept.
- Connecting a question with previous notes.
- Continuing a multi-turn discussion about the same book.

In the multi-agent system, DeepReadAgent is positioned between upstream control agents and downstream review or post-processing agents. The upstream components include the input safety filter and RouterAgent. The downstream agents include CriticAgent, NoteAgent, and FollowupAgent.

### 3.2 Reasoning and Planning Workflow

DeepReadAgent follows a tool-using reasoning workflow based on the ReAct pattern. Instead of directly generating an answer, the agent can decide to call retrieval tools first. This allows the response to be grounded in relevant book evidence.

The simplified workflow is:

1. Receive the user query and contextual information from the API layer.
2. Build a dynamic system context using the current book, selected text, current chapter, and memory context.
3. Use the retrieval tool to search relevant book content.
4. Optionally search the user's past notes when the question benefits from previous reading context.
5. Generate an answer based on the evidence returned by the tools.
6. Return the answer with citations.
7. Stream the answer tokens to the frontend when the streaming endpoint is used.

This workflow ensures that the agent's reasoning is not isolated from the reading material. The agent is designed to behave as a deep reading assistant rather than a general-purpose chatbot.

### 3.3 DeepReadAgent Workflow

The DeepReadAgent workflow connects user interaction, safety control, retrieval, generation, and post-processing. The following workflow summarises how the agent collaborates with the wider Kant system:

| Step | Component | Responsibility |
|---|---|---|
| 1 | User / Frontend | User asks a book-related question, optionally with selected text and current chapter context. |
| 2 | Security Filter | Blocks prompt injection, secret leakage, unsafe file access, or command execution requests. |
| 3 | RouterAgent | Identifies whether the request is a book question and prepares the query for the answer workflow. |
| 4 | DeepReadAgent | Builds dynamic context and decides whether to call retrieval tools. |
| 5 | `search_book_content` | Retrieves relevant book evidence using hybrid retrieval. |
| 6 | `search_past_notes` | Retrieves previous user notes when relevant. |
| 7 | LLM Generation | Produces an answer grounded in retrieved evidence. |
| 8 | Citation Builder | Returns citations from the retrieved documents. |
| 9 | CriticAgent / NoteAgent / FollowupAgent | Reviews the answer, saves useful notes, and generates reflective follow-up questions. |

### 3.4 Memory and State Management

DeepReadAgent uses state management to support multi-turn reading conversations. Short-term conversation history is handled through LangGraph's SQLite checkpointer. A deterministic thread identifier is constructed from the user ID and book ID, allowing the system to maintain separate conversations for different users and books.

The agent can also receive long-term memory context from the wider system. This context may include previous reading preferences or relevant historical interactions. In the implementation, the memory context is injected into the dynamic system message rather than mixed directly into the user's question. This design helps separate user intent from system-provided context and makes the prompt structure easier to control.

### 3.5 Tools Integrated

DeepReadAgent uses scoped tools that are directly related to reading and retrieval. The main tools are:

- `search_book_content`: searches the current book or library content for textual evidence.
- `search_past_notes`: searches the user's previous notes for relevant concepts or ideas.

The `search_book_content` tool is connected to the RAG retrieval layer. It uses a hybrid retrieval approach that combines keyword-based retrieval and vector-based retrieval. ChromaDB is used for vector search, while BM25-style keyword retrieval improves matching for exact terms, names, and phrases. The retrieval results are then used to build evidence blocks that the language model can use when generating an answer.

The `search_past_notes` tool connects the agent with the user's note-taking workflow. It allows DeepReadAgent to refer to previous reading notes without giving the agent unrestricted access to the file system.

### 3.6 Communication and Coordination Logic

DeepReadAgent is not an isolated component. It communicates with other parts of Kant through the backend orchestration flow:

- The input safety layer checks the user's query before it reaches the agent.
- RouterAgent classifies the intent and can pass an optimised query to DeepReadAgent.
- DeepReadAgent retrieves evidence and generates the main response.
- CriticAgent reviews the generated answer for hallucination, objectivity, and reliability.
- NoteAgent can extract useful insights from the answer and save them into the note system.
- FollowupAgent generates reflective follow-up questions based on the user's question and the answer.

This coordination pattern allows DeepReadAgent to focus on its core responsibility: evidence-grounded reading answers. Other agents handle routing, review, note generation, and follow-up support.

### 3.7 Prompt Engineering Patterns Used

Several prompt engineering patterns are used in DeepReadAgent:

- **Dynamic system context:** current book, book source, selected text, current chapter, and memory context are assembled into the system prompt.
- **Evidence-grounded instruction:** the agent is instructed to base answers on retrieved book evidence.
- **Tool-use prompting:** the agent is given retrieval tools and can call them before answering.
- **Locale-aware prompts:** the prompt bundle supports different languages, such as Chinese and English.
- **Context separation:** system-provided context is separated from the raw user query to reduce confusion and improve controllability.

These patterns help the agent maintain a clear role, use tools appropriately, and produce answers that are more traceable and useful for reading tasks.

### 3.8 Fallback Strategies

DeepReadAgent includes several fallback-oriented design choices:

- If no strong evidence is retrieved, the agent should avoid inventing unsupported details.
- If long-term memory is unavailable, the agent can still answer based on the current book content.
- If the external safety service is unavailable, the system can fall back to local rule-based filtering before the request reaches the agent.
- If no past notes are found, the agent can continue using book content retrieval only.

These fallback strategies improve system robustness and reduce dependency on any single external component.

## 4. Implementation Details

### 4.1 Summary of Implementation Approach

DeepReadAgent is implemented in the backend as a Python class. It is integrated into the FastAPI chat workflow and supports both normal response mode and streaming response mode.

The implementation uses LangGraph's `create_react_agent` to construct a tool-using agent. Each request builds a fresh agent instance with tool closures bound to the current request context. This design reduces the risk of cross-user state contamination and makes the agent safer for concurrent use.

The main runtime input fields include:

- `query`
- `book_source`
- `book_id`
- `memory_context`
- `user_id`
- `thread_id`
- `selected_text`
- `current_chapter`
- `locale`

These fields allow the agent to respond to the user's question in the context of a specific book and reading session.

### 4.2 Code Structure Overview

The main implementation files related to DeepReadAgent are:

- `backend/agents/deepread_agent.py`: main DeepReadAgent implementation, configuration, tool construction, streaming logic, and chat history functions.
- `backend/api/chat.py`: API orchestration layer that calls DeepReadAgent and coordinates safety checks, routing, review, notes, and follow-up generation.
- `backend/rag/retriever/hybrid_retriever.py`: hybrid retrieval implementation used by the book search tool.
- `backend/rag/chroma/chroma_store.py`: ChromaDB vector storage and retrieval support.
- `backend/xai/citation.py`: citation construction for explainability.
- `backend/security/input_filter.py`: safety filtering before the request reaches the agent.

In `deepread_agent.py`, `DeepReadConfig` controls the retrieval configuration, such as the number of retrieved documents and maximum evidence blocks. `DeepReadResult` stores the final answer, citations, and retrieved documents. The `run()` method is used for non-streaming responses, while `astream_events()` is used for server-sent event streaming.

### 4.3 Tech Stack of the Agent

The DeepReadAgent implementation uses the following technologies:

- Python 3.12 in the project `.venv312` environment.
- FastAPI for backend API integration.
- LangGraph and LangChain for agent construction and tool execution.
- OpenAI-compatible LLM API for answer generation.
- ChromaDB for vector-based retrieval.
- BM25-style retrieval for keyword matching.
- SQLite for LangGraph checkpointer-based conversation state.
- Server-Sent Events for streaming responses to the frontend.

This stack supports both experimentation and deployment-oriented engineering. The design is modular, so retrieval, prompting, citation construction, and API orchestration can be tested and improved separately.

## 5. Key Design Decisions

Several design decisions were important for making DeepReadAgent suitable for a production-oriented agentic AI system.

| Design Decision | Reason | Benefit |
|---|---|---|
| Use a ReAct tool-using agent instead of a single LLM call | A single LLM call cannot reliably inspect the uploaded book content. | The agent can actively retrieve evidence before generating an answer. |
| Keep retrieval tools scoped | DeepReadAgent only needs reading-related capabilities. | Reduces the risk of unsafe autonomous behaviour. |
| Inject reading context into the system message | Current book, chapter, selected text, and memory context are system-provided context rather than raw user instructions. | Improves prompt control and reduces ambiguity between user intent and system context. |
| Return citations as part of the agent result | Reading answers should be verifiable. | Supports explainability and user trust. |
| Use deterministic thread IDs based on user and book | Different users and books need separate conversation state. | Prevents unrelated conversations from mixing. |
| Support streaming responses | Long reading answers may take time to generate. | Improves perceived responsiveness in the frontend. |

These decisions show that the agent was designed with engineering boundaries, user experience, and safety in mind.

## 6. Testing and Validation

### 6.1 Types of Tests Implemented

The testing strategy for DeepReadAgent is connected to the wider backend test suite. The relevant tests include unit tests, integration tests, and AI security tests.

**Unit tests** focus on lower-level components that DeepReadAgent depends on:

- Text cleaning and chunking.
- ChromaDB storage and retrieval.
- Hybrid retrieval behaviour.
- Citation construction.
- Agent-related utility behaviour.

**Integration tests** validate that the agent works correctly within the backend workflow:

- Chat API request handling.
- Book-specific retrieval.
- Response generation with citations.
- Streaming response events such as token, tool status, and completion events.

**AI security tests** check whether malicious or unsafe inputs are handled correctly:

- Prompt injection attempts.
- Requests to reveal secrets or API keys.
- Requests to access or modify local files.
- Requests to execute commands.
- Adversarial inputs that attempt to bypass the intended reading assistant role.

### 6.2 Results and Key Findings

The tests showed that the backend architecture supports the main DeepReadAgent workflow: user questions can pass through the API layer, retrieve relevant book evidence, generate an answer, and return citations. The security tests also validate that unsafe requests can be blocked before reaching the agent.

Important findings from validation include:

- Retrieval quality is central to answer quality. If evidence is weak or missing, the answer should avoid unsupported claims.
- Citations are necessary for user trust because they allow the reader to verify the generated answer.
- Streaming improves user experience, but it requires careful event handling so that tool status, tokens, and completion metadata are returned correctly.
- Security filtering must happen before agent execution because an LLM agent may otherwise be exposed to malicious user instructions.

The project also includes LLMSecOps evidence such as backend test results, coverage reports, AI security test evidence, dependency scanning, SAST scanning, and container scan evidence. These artifacts support the reliability and security claims made for the system.

### 6.3 Testing Evidence Mapping

| Test or Evidence Type | What It Validates | Related Project Requirement |
|---|---|---|
| Retriever tests | The system can retrieve relevant book chunks for evidence-grounded answers. | Explainable AI and RAG reliability |
| Chroma store tests | Vector storage and retrieval functions work as expected. | System integration and data layer reliability |
| Chat API tests | DeepReadAgent can be triggered through the backend API workflow. | Agent integration and deployment readiness |
| Streaming behaviour checks | Token, tool, and completion events can be sent to the frontend. | User-facing system integration |
| AI security tests | Prompt injection, secrets, filesystem, and command execution requests are blocked. | AI Security |
| Coverage report | Core backend modules are covered by automated tests. | LLMSecOps quality control |
| Snyk and SAST evidence | Dependency and code-level risks are scanned. | MLSecOps / LLMSecOps governance |

This mapping connects the agent-level validation work with the assessment requirements. It also shows that DeepReadAgent is not only implemented but tested as part of a controlled engineering pipeline.

## 7. Explainable and Responsible AI Considerations

### 7.1 Explainability

Explainability is one of the most important design goals of DeepReadAgent. In a reading system, the user should be able to understand why the AI produced a particular answer and where the supporting information came from. DeepReadAgent addresses this by returning citations together with the generated answer.

The citations are built from retrieved documents and can include source metadata such as the book title, chapter or section information, and a supporting snippet. This allows users to compare the answer with the original text. As a result, the agent's output is not only a fluent response but also an auditable reading interpretation.

### 7.2 Responsible AI Alignment

DeepReadAgent is designed to avoid unsupported generation. Since the agent uses retrieval tools before answering, it is encouraged to ground its response in available book evidence. This is especially important for literary, philosophical, or conceptual reading questions where an answer may otherwise become too speculative.

The agent also fits into a broader responsible AI workflow. CriticAgent reviews the generated answer after the main response. This post-hoc review helps identify possible hallucination, weak evidence, or overly subjective claims. In addition, the system treats uploaded books and retrieved text as untrusted content rather than instructions to be followed.

### 7.3 Bias Mitigation Strategies

DeepReadAgent does not perform high-stakes classification, but bias can still appear in interpretation and summarisation. To reduce this risk, the agent is designed to prioritise textual evidence over unsupported assumptions. For open-ended interpretation questions, the answer should present balanced reasoning and avoid claiming a single absolute interpretation when the text supports multiple readings.

The combination of evidence retrieval, citations, and CriticAgent review helps reduce the risk of biased or overconfident responses.

### 7.4 Handling Sensitive Content and Governance Alignment

Governance is supported through several workflow-level controls. User input is checked before it reaches DeepReadAgent. Tool access is limited to reading-related retrieval functions. The agent does not receive tools for arbitrary code execution or unrestricted local file access. Generated answers and retrieval behaviour can be monitored through logs and testing evidence.

This design aligns with responsible AI principles because it keeps the agent's authority bounded, makes outputs traceable, and supports ongoing monitoring.

## 8. Security Practices

### 8.1 Agent-Specific Security Risks

DeepReadAgent faces several security risks:

- **Prompt injection from user input:** a user may ask the agent to ignore previous instructions or reveal hidden prompts.
- **Prompt injection from retrieved content:** malicious text inside an uploaded book could attempt to influence the agent's behaviour.
- **Hallucination:** the agent may generate claims that are not supported by retrieved evidence.
- **Sensitive information leakage:** the agent may be asked to reveal API keys, secrets, or internal system details.
- **Over-broad tool access:** if the agent had unrestricted tools, it could perform actions beyond the reading assistant scope.
- **Adversarial inputs:** carefully written inputs may attempt to bypass role boundaries or safety controls.

### 8.2 Mitigations Implemented

Several mitigations are used at the workflow and code levels:

- User input is checked by Lakera Guard when configured.
- If the external safety service is unavailable, the system falls back to local regular-expression based safety rules.
- DeepReadAgent only exposes scoped retrieval tools, not arbitrary filesystem or code execution tools.
- Book retrieval can be filtered by book source, reducing irrelevant cross-book evidence.
- The agent is prompted to produce evidence-grounded answers.
- Citations are returned so that the user can verify the answer.
- CriticAgent performs a post-answer review for hallucination and objectivity.
- Security tests are included in the backend test suite and CI/CD pipeline.

These controls reduce the risk that DeepReadAgent will behave as an uncontrolled autonomous agent. The agent is powerful enough to retrieve and reason over reading materials, but its actions remain bounded by the reading workflow.

### 8.3 Agent-Specific Security Risk Register

| Security Risk | Possible Impact | Mitigation | Validation |
|---|---|---|---|
| Prompt injection from user input | The agent may be instructed to ignore its role or reveal hidden instructions. | Lakera Guard and local safety fallback run before agent execution. | AI security tests for prompt injection patterns. |
| Prompt injection from retrieved book content | Malicious book text may attempt to influence the answer. | Retrieved text is treated as evidence, not as system instruction; tools are scoped. | Manual review and future RAG injection tests. |
| Hallucination | The agent may generate unsupported claims. | Hybrid retrieval, evidence-grounded prompting, citations, and CriticAgent review. | Answer and citation validation. |
| Sensitive information leakage | User or system secrets may be exposed. | Secret patterns are blocked by the input safety layer. | Security tests for API key and token patterns. |
| Over-broad tool access | Agent may perform unsafe actions outside reading scope. | DeepReadAgent only receives retrieval and note-search tools. | Code review of exposed tools. |
| Weak or irrelevant retrieval | Poor evidence may reduce answer quality. | Hybrid retrieval and metadata filtering by book source. | Retriever tests and manual answer inspection. |

This risk register is specific to DeepReadAgent and shows how security was considered at the agent workflow level rather than only at the infrastructure level.

## 9. Reflection

Implementing DeepReadAgent helped me understand that an agentic AI system is not simply an LLM with a prompt. A reliable agent requires tool design, retrieval quality, state management, safety controls, explainability, and testing. The most important lesson for me was that good agent design is about controlling the flow of information and action, not only improving the final generated text.

One major challenge was balancing response quality with evidence grounding. Users expect natural and insightful reading explanations, but the system must avoid making claims that cannot be supported by the book. The citation mechanism and retrieval workflow helped address this issue by making the answer traceable.

Another important learning point was the importance of scoped tool access. DeepReadAgent only needs tools for book content retrieval and note search. Giving the agent broader capabilities would increase security risk without improving the core reading workflow.

For future improvement, I would focus on:

- More fine-grained citations at paragraph or page level.
- A dedicated RAG evaluation benchmark for answer faithfulness and retrieval quality.
- Stronger detection of prompt injection inside retrieved book content.
- Better confidence indicators when evidence is weak.
- More detailed tracing for latency, token usage, retrieval quality, and cost monitoring.

Overall, DeepReadAgent demonstrates how an AI reading assistant can be designed as a controlled, explainable, and security-aware agent rather than a general chatbot. It contributes to the Kant system by providing the main evidence-grounded reasoning capability required for deep reading.
