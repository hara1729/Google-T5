import wikipedia


class WikipediaRetriever:
    """
    Robust wrapper around `wikipedia` that

    * walks through all disambiguation options until one loads,
    * falls back to `wikipedia.summary` if every page fails, and
    * never throws - returns "" on any unrecoverable error.
    """

    def __init__(self, num_sentences: int = 3, max_tries: int = 5):
        self.num_sentences = num_sentences
        self.max_tries = max_tries

    def _try_page(self, title: str) -> str | None:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError:
            return None
        except Exception:
            return None

    def fetch_context(self, query: str) -> str:
        # 1) direct attempt
        text = self._try_page(query)
        if text:
            return " ".join(text.split(". ")[: self.num_sentences])

        # 2) walk through disambiguation options
        try:
            options = wikipedia.search(query)[: self.max_tries]
            for cand in options:
                text = self._try_page(cand)
                if text:
                    return " ".join(text.split(". ")[: self.num_sentences])
        except Exception:
            pass  # ignore network / http errors

        # 3) last fallback: short summary API
        try:
            summary = wikipedia.summary(query, sentences=self.num_sentences, auto_suggest=False)
            return summary
        except Exception:
            return ""  # give up gracefully
