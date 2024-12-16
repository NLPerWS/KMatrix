from infer import infer
# from embedding import embedding
from root_config import RootConfig
import os


if __name__ == "__main__":
    test_case = [
        {
            "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
            "title": "\"Algorithms for calculating variance\"\n",
            "content": "Moonda is over here. In fact, he is monitor.",
            "dataframe": None,
            "blob": None,
            "score": None,
            "embedding": None,
            "BGE_embedding": None,
            "contriever_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "DPR_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "header": None,
            "rows": None,
            "triples": None,
            "source": "表格: wikitable"
        },
        {
            "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
            "title": "This is a test title.",
            "content": "ChatGPT can help you anytime, anywhere.",
            "dataframe": None,
            "blob": None,
            "score": None,
            "embedding": None,
            "BGE_embedding": None,
            "contriever_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "DPR_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "header": None,
            "rows": None,
            "triples": None,
            "source": "表格: wikitable"
        },
        # {
        #     "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
        #     "content": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0\tpat quinn (politician) was 41st governor of illinois from january 29, 2009 – january 12, 2015. pat quinn (politician) was the 45th lieutenant governor of illinois. sheila simon was preceded by rod blagojevich and succeeded by bruce rauner. pat quinn (politician) was governor from january 13, 2003 – january 29, 2009 under governor rod blagojevich. corinne wood preceded pat quinn. pat quinn (politician) was 70th treasurer of illinois from january 14, 1991 – january 9, 1995 and was succeeded by sheila simon. pat quinn (politician) was commissioner of the cook county board of appeals when jim edgar was governor. jerome cosentino was preceded by pat quinn (politician) and judy baar topinka was succeeded by jim edgar.\tPat Quinn (politician)",
        #     "dataframe": None,
        #     "blob": None,
        #     "score": None,
        #     "embedding": None,
        #     "BGE_embedding": [0.0002827, -0.0008128],
        #     "contriever_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "DPR_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "header": None,
        #     "rows": None,
        #     "triples": None,
        #     "source": "表格: wikitable"
        # },
        # {
        #     "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
        #     "content": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0\tpat quinn (politician) was 41st governor of illinois from january 29, 2009 – january 12, 2015. pat quinn (politician) was the 45th lieutenant governor of illinois. sheila simon was preceded by rod blagojevich and succeeded by bruce rauner. pat quinn (politician) was governor from january 13, 2003 – january 29, 2009 under governor rod blagojevich. corinne wood preceded pat quinn. pat quinn (politician) was 70th treasurer of illinois from january 14, 1991 – january 9, 1995 and was succeeded by sheila simon. pat quinn (politician) was commissioner of the cook county board of appeals when jim edgar was governor. jerome cosentino was preceded by pat quinn (politician) and judy baar topinka was succeeded by jim edgar.\tPat Quinn (politician)",
        #     "dataframe": None,
        #     "blob": None,
        #     "score": None,
        #     "embedding": None,
        #     "BGE_embedding": [0.0002827, -0.00128],
        #     "contriever_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "DPR_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "header": None,
        #     "rows": None,
        #     "triples": None,
        #     "source": "表格: wikitable"
        # },
        # {
        #     "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
        #     "content": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0\tpat quinn (politician) was 41st governor of illinois from january 29, 2009 – january 12, 2015. pat quinn (politician) was the 45th lieutenant governor of illinois. sheila simon was preceded by rod blagojevich and succeeded by bruce rauner. pat quinn (politician) was governor from january 13, 2003 – january 29, 2009 under governor rod blagojevich. corinne wood preceded pat quinn. pat quinn (politician) was 70th treasurer of illinois from january 14, 1991 – january 9, 1995 and was succeeded by sheila simon. pat quinn (politician) was commissioner of the cook county board of appeals when jim edgar was governor. jerome cosentino was preceded by pat quinn (politician) and judy baar topinka was succeeded by jim edgar.\tPat Quinn (politician)",
        #     "dataframe": None,
        #     "blob": None,
        #     "score": None,
        #     "embedding": None,
        #     "BGE_embedding": [0.01827, -0.00128],
        #     "contriever_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "DPR_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "header": None,
        #     "rows": None,
        #     "triples": None,
        #     "source": "表格: wikitable"
        # },
        # {
        #     "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_1",
        #     "content": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_1\tpat quinn (politician) was born patrick joseph quinn, jr. december 16, 1948 (age 71) chicago, illinois, u.s. pat quinn (politician) was in office 1982–1986. pat quinn (politician) is a democratic. his spouse(s) is julie hancock (m. 1982; div. 1986). he attended georgetown university (bs) and northwestern university (jd). he has 2 sons.\tPat Quinn (politician)",
        #     "dataframe": None,
        #     "blob": None,
        #     "score": None,
        #     "embedding": None,
        #     "BGE_embedding": [0.01827, -0.0028],
        #     "contriever_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "DPR_embedding": [
        #         [
        #             0.1
        #         ],
        #         [
        #             0.2
        #         ]
        #     ],
        #     "header": None,
        #     "rows": None,
        #     "triples": None,
        #     "source": "表格: wikitable"
        # },
    ]
    
    test_do_embedding = [
        {
            "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
            "title": "\"Algorithms for calculating variance\"\n",
            "text": "Moonda is over here. In fact, he is monitor.",
            "dataframe": None,
            "blob": None,
            "score": None,
            "embedding": None,
            "BGE_embedding": None,
            "contriever_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "DPR_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
        },
        {
            "id": "ott-wiki-table-v3:_Pat Quinn (politician)_F85BEFF5891F2E08_0",
            "title": "This is a test title.",
            "text": "ChatGPT can help you anytime, anywhere.",
            "dataframe": None,
            "blob": None,
            "score": None,
            "embedding": None,
            "BGE_embedding": None,
            "contriever_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
            "DPR_embedding": [
                [
                    0.1
                ],
                [
                    0.2
                ]
            ],
        },
    ]
    
    infer(
        model=RootConfig.DPR_model_path,
        input_query=['Who is moonda?'], 
        passage=test_case,
        top_k=3,
    )
