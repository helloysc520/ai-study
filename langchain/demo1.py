import tiktoken

encoding = tiktoken.get_encoding('cl100k_base')

res = encoding.encode('tiktoken is great!')
print(res)


def num_tokens_from_string(string:str,encoding_name:str) -> int:

    enconding = tiktoken.get_encoding(encoding_name)

    num_tokens = len(enconding.encode(string))

    return num_tokens

res = num_tokens_from_string('tiktoken is great!','cl100k_base')
print(res)

res = encoding.decode([83, 1609, 5963, 374, 2294, 0])
print(res)

res = [encoding.decode_single_token_bytes(token) for token in [83, 1609, 5963, 374, 2294, 0]]
print(res)