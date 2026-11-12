Александр, спасибо за разбор. Черновик пока не правим, как ты и предложил. Ниже — что значили формулировки, которые звучат «слишком по-блоговски» или неочевидно. В скобках — более простой английский.

---

**1. Название и строка «First draft…»**

Заголовок сейчас: *Getting Milvus nearest-neighbour lookup to run on the 7900 XTX*
- По-русски: «Как мы заставили Milvus искать ближайших соседей (векторы) на видеокарте 7900 XTX».
- Simpler English: “How we made Milvus vector search work on an RX 7900 XTX.”
- Почему так: nearest-neighbour lookup = поиск ближайших векторов (ANN), не «поиск в Google». 7900 XTX = конкретная Radeon, на которой мы работали.

Строка *First draft. One piece of text, three places: Medium, LinkedIn, AMD blog if they take it.*
- Это не для читателя. Это пометка для нас: один текст на три площадки, AMD блог — если примут.
- Согласен: в публичной версии её не будет. (We will delete this line before publishing.)

**2. Зачем Milvus и I → we**

*I needed a RAG database on the box I already had.*
- «Мне нужна была векторная БД для RAG на том компьютере, который уже стоял.»
- Simpler: “We needed a vector database for RAG on the PC we already had.”

*I wanted Milvus: it is the most popular open-source one for that (~46k GitHub stars).*
- «Мы взяли Milvus, потому что это самая популярная открытая векторная БД для RAG (около 46 тыс. звёзд на GitHub).»
- Analogues: аналоги были (FAISS — библиотека, pgvector — кусок Postgres, облака). Мы не говорим «других не было». Мы говорим: для RAG со GPU-индексами люди ставят именно Milvus; его GPU-путь только NVIDIA.
- Реальная задача «зачем»: в черновике пока нет имени продукта — только честная рамка (векторный поиск / RAG на той машине, что была). Если есть внутреннее имя проекта — можно подставить. Иначе в следующей версии: «нужен был GPU nearest-neighbour search в том стеке, который ставят в RAG».

I → we: ок, в следующей версии от команды («we»), не «я купил карту». (We will write as the team, not as one person.)

**3. Фразы, которые режут слух (это разговорный американский, не «ошибка Grok» как таковая, но для техблога/AMD они слишком устные)**

*Python never talks to cuVS*
- Буквально: «Python никогда не разговаривает с cuVS».
- Имелось в виду: клиент на Python не вызывает GPU-библиотеку напрямую. Цепочка: Python → сервер Milvus → Knowhere → cuVS.
- Simpler: “The Python client does not call cuVS directly.”

*I still trip over that when I switch machines*
- «Я до сих пор об это спотыкаюсь, когда меняю компьютер» (идиома trip over = забываю / наступаю на те же грабли).
- Simpler: “I still make this mistake when I use another PC.” (Forgot: GPU on AMD must be built from source, not Docker.)

*We didn't do a weekend find-and-replace of cuVS for hipVS*
- «Мы не заменили cuVS на hipVS за выходные поиском-и-заменой по тексту.»
- Simpler: “This was not a two-day search-and-replace. We did it in steps.”

*Layer 1.5 still annoys me*
- «Этап 1.5 меня до сих пор бесит» (warp 32 vs 64, пустые соседи).
- Simpler: “Step 1.5 was painful. The IVF packer used the wrong warp size.”

Для AMD-блога такие фразы лучше снять: у них в гайде как раз «conversational, но простой и короткий», не сленг. Ссылка: https://github.com/ROCm/rocm-blogs/wiki

**4. *Two different ports, not one story***

- «Два разных порта, это не одна история» — заголовок в стиле слогана.
- Имелось в виду: работа Кости (помощник в IDE) и работа Максима (замкнутый агентный пайплайн) — два метода, не один.
- Simpler heading later: “We used agents in two ways” / «Два способа работы с агентами».

**5. RTX 4080 и “people here said”**

*didn't have an RTX 4080 at the beginning*
- «Сначала у нас не было RTX 4080.»
- Согласен: сравнение с NVIDIA ближе к графикам, не в первом экране. (Move the NVIDIA PC to the results section.)

*people here said I should measure*
- «Коллеги сказали, что надо измерить.» Слишком «рассказик». Уберём. (We will delete this.)

**6. Скриншоты слайдов**

Согласен: в статью — только графики (оси, линии), без рамки DXC-слайда. (Crop to the chart only.)

---

**Как идём дальше (твоё предложение)**

Черновик v1 не переписываем построчно. Сначала человеческая структура будущего текста (не generated plan). Потом черновик по главам, можно через Claude/GPT. Для AMD — стиль из wiki: читатель сначала, короткий язык, we/active voice, техника, не продажа, обрезанные рисунки, публикация через issue/PR в ROCm/rocm-blogs.

План статьи на русском — отдельным файлом, пришлю следом.