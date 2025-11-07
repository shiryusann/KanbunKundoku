import spacy
from typing import List, Tuple, Optional, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

pos_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは日本語の専門家です。次は漢文、書き下し文及び書き下し文の品詞タグ付きです。これに基づき、書き下し文の品詞タグを訂正してください。\
                    GiNZA品詞体系を使ってください。動詞と名詞の語幹及び送り仮名をに注意してください。助詞、例えば「て」、助動詞、例えば「し」に注意してください。\
                    再読文字、例えば「未だ」「将に」「須らく」「若お」「蓋ぞ」、とそれらの送り仮名に注意してください。形式名詞に対して、「名詞-普通名詞-形式名詞」のタグを使ってください。"),
        MessagesPlaceholder("chinese_sentence"),
        MessagesPlaceholder("japanese_sentence"),
        MessagesPlaceholder("tagged_sentence"),
    ]
)

class POS(BaseModel):
    characters: Optional[str] = Field(default = None, description = "Japanese kanji and kana belonging to a single word")
    pos: Optional[str] = Field(default = None, description = "English tag of the specificed word")
    tag: Optional[str] = Field(default = None, description = "Japanese tag of the specificed word")

class POSs(BaseModel):
    poss: List[POS] = Field(default = None, description = "the list of pos tags for each word")

class spacyPOSTool(BaseTool):
    name:str = "pos_tool"
    description:str = "part of speech tagging using spacy model"
    return_direct:bool = True
    nlp:ClassVar[spacy.language] = spacy.load("ja_ginza")

    def _run(self, sentence:str):
        tags = []
        doc = self.nlp(sentence)
        for token in doc:
            tags.append((token.text, token.pos_, token.tag_))
        return tags

class llmPosReponseParseTool(BaseTool):
    name:str = "parse_tool"
    description:str = "get the response of llm and extract structured data"
    return_direct:bool = True

    llm:BaseChatModel

    def _run(self, chinese_sentence, japanese_sentence:str, tokens:List[Tuple[str, str, str]]):
        structured_llm = self.llm.with_structured_output(schema = POSs)
        prompt = pos_prompt_template.invoke(
            {
                "chinese_sentence":[("user", "漢文\n" + chinese_sentence)],
                "japanese_sentence": [("user", "書き下し文\n" + japanese_sentence)],
                "tagged_sentence": [("user", "品詞タグ\n" + "\n".join(["\t".join(token) for token in tokens]))]
            }
        )
        return structured_llm.invoke(prompt)