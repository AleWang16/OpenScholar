For run_chunk_and_embed.py, I want you to make a modification.  Notice in the processed_json folder that there are two json files.  Their formats do not correspond to the the desired format for the input files.  Notice that there is a field called ref_en
tries, where it lists every table or figure in the json, and they are used in the json as follows:

{"paper_id": "baumert2014",
    "header": {
        "generated_with": "S2ORC 1.0.0",
        "date_generated": "2026-02-18T15:59:55.135117Z"
    },
    "title": "SOIL ORGANIC CARBON SEQUESTRATION IN JATROPHA CURCAS SYSTEMS IN BURKINA FASO",
    "authors": [
        {
            "first": "Sophia",
            "middle": [],
            "last": "Baumert",
            "suffix": "",
            "affiliation": {},
            "email": "sbaumert@uni-bonn.de"
        },
        {
            "first": "Asia",
            "middle": [],
            "last": "Khamzina",
            "suffix": "",
            "affiliation": {},
            "email": ""
        },
        {
            "first": "Paul",
            "middle": [
                "L G"
            ],
            "last": "Vlek",
            "suffix": "",
            "affiliation": {},
            "email": ""
        }
    ],
    "year": "",
    "venue": null,
    "identifiers": {},
    "pdf-parse": {
        "paper_id": "baumert2014",
    "abstract": [
            {
                "text": ---text here
                "cite_spans": [],
                "ref_spans": [],
                "eq_spans": [],
                "section": "Abstract",
                "sec_num": null
            }
        ],
        "body_text": [
            {
                "text": --text here
                "ref_spans": [
                    {
                        "start": 167,
                        "end": 168,
                        "text": "1",
                        "ref_id": "TABREF0"
                    }
                ],
            },
            {
                "text": --text here
            }
        ]
        "ref_entries": [
            TABREF0: {
                
                    content: --table content here
            }
        ]

    }
}

Notice that this is a nonlinear way of parsing the paper, because tables are referenced in the body text, but the 
full content is in the ref-entries section.  I want you to splice this such that the table and the body text are 
concatentated with each other such that the paper reads linearly, as originally intended, but keep things such as the paper 
id, title and authors in the same format. Here is an example below:

{"paper_id": "baumert2014",
    "header": {
        "generated_with": "S2ORC 1.0.0",
        "date_generated": "2026-02-18T15:59:55.135117Z"
    },
    "title": "SOIL ORGANIC CARBON SEQUESTRATION IN JATROPHA CURCAS SYSTEMS IN BURKINA FASO",
    "authors": [
        {
            "first": "Sophia",
            "middle": [],
            "last": "Baumert",
            "suffix": "",
            "affiliation": {},
            "email": "sbaumert@uni-bonn.de"
        },
        {
            "first": "Asia",
            "middle": [],
            "last": "Khamzina",
            "suffix": "",
            "affiliation": {},
            "email": ""
        },
        {
            "first": "Paul",
            "middle": [
                "L G"
            ],
            "last": "Vlek",
            "suffix": "",
            "affiliation": {},
            "email": ""
        }
    ],
    "year": "",
    "venue": null,
    "identifiers": {},
    "text": --text and TABREF content all in one place
}


So ideally, you create a function, where you take a json in the former format, and convert it to the latter. 