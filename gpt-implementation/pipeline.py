import tiktoken

def main():
    with open('input.txt', 'r', encoding="utf-8") as f:
        text = f.read()

    print("Length of dataset in characters: ", len(text))
    print(text[:1000])

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print("".join(chars))
    print(vocab_size)

    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]
    def decode(ls):
        return "".join([itos[i] for i in ls])

    print(encode("hi there"))
    print(decode(encode("hi there")))

    enc = tiktoken.get_encoding("gpt2")
    print(enc.decode(enc.encode("hello world")))


if __name__ == "__main__":
    main()

