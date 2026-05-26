# 使用技巧与注意事项

> 英文版：[tips.md](../tips.md)

- **`ref_audio` 与 `instruct` 的组合**：
  当同时提供 `ref_audio` 和 `instruct` 且二者**冲突**时，模型通常会跟随参考音频的风格。当二者**一致**时，`instruct` 可以改善其所描述属性的克隆稳定性。典型例子是**中文方言克隆**：同时提供方言参考音频和匹配的方言 instruct（例如 `ref_audio="sichuan.wav", instruct="四川话"`），方言输出会更稳定。

- **短音频生成**：
  在没有参考音频时，模型可能无法可靠地生成很短（例如 1–2 秒）的片段。若需要短片段，请向模型提供参考音频。

- **闽南语输入格式**：
  闽南语（Min Nan / Hokkien）当前版本只能使用 [台罗拼音（Tâi-lô）](https://en.wikipedia.org/wiki/T%C3%A2i-l%C3%B4) 作为输入进行合成，不支持汉字输入。
