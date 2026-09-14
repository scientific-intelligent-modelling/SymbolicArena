# Reviewer Model API Smoke Test

本目录记录将 Opus5 替换为其它模型执行公式化简复核前的最小非流式 API 测试。

## 测试条件

- API channel: Routify Anthropic-compatible Messages API
- stream: false
- temperature: 0
- max output tokens: 800
- credential: 只从本机 profile 读取，审计文件中已脱敏
- cases:
  - A: `x+x -> 2*x` over real x，应通过；
  - B: `sqrt(x**2) -> x` over real x，应失败；
  - C: gplearn `protected_sqrt(z)=sqrt(abs(z)) -> sqrt(z)`，应失败。

## 结果

| Model | HTTP | Attempts | Strict JSON | Correct | Input tokens | Output tokens | Usable |
|---|---:|---:|---|---:|---:|---:|---|
| kimi-k3 | 200 | 1 | passed | 3/3 | 313 | 211 | yes |
| glm-5.2 | 200 | 1 | passed | 3/3 | 154 | 141 | yes |
| gpt-5.5 | 400 | 2 | no response | 0/3 | 0 | 0 | no deployment |
| gpt-5.6-sol (Anthropic endpoint) | 400 | 2 | no response | 0/3 | 0 | 0 | wrong endpoint |
| gpt-5.6-sol (OpenAI Responses) | 200 | 1 | passed | 3/3 | 224 | 141 | yes |

`gpt-5.5` 和 `gpt-5.6-sol` 发送到 Anthropic-compatible 端口时都返回 `NoAvailableModels`。将 `gpt-5.6-sol` 改发至 `/protocol/openai/v1/responses` 后，第一次请求即成功，说明此前失败是协议端口错误，不是模型未部署或能力不足。

## 结论

`kimi-k3`、`glm-5.2` 和 OpenAI Responses 端口上的 `gpt-5.6-sol` 均可进入下一轮真实 Opus5 化简结果校准。此次 smoke 只证明接口可用、严格 JSON 可解析并能识别三个基础语义陷阱，不代表已经验证大规模公式评审质量。

使用此前 Opus5 的输入 3 元/百万 token、输出 15 元/百万 token 作为保守参考，三次成功请求对应约 0.009468 元；不同模型的真实账单价格仍应以渠道结算为准。
