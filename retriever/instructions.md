For run_chunk_and_embed.py, I want you to make a modification.  Notice in the processed_json folder that there are two json files.  Their formats do not correspond to the the desired format for the input files.  Notice that there is a field called ref_entries, where it lists every table or figure in the json, and they are used in the json as follows:

{
    pdf-parse:
        body-text: [
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
            }
        ]
        ref-entires: [
            TABREF0: {
                
                    content: --table content here
            }
        ]
}

Notice that this is a nonlinear way of parsing the paper, because tables are referenced in the body text, but the 
full content is in the ref-entries section.  I want you to splice this such that the table and the body text are 
concatentated with each other such that the paper reads linearly, as originally intended. Here is an example below:

{
    text: --text and table content here
}

The input and output would both be a json.  