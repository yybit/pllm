.PHONY: testdata
testdata: \
	testdata/stories15M.bin

testdata/stories15M.bin:
	curl -o $@ -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
