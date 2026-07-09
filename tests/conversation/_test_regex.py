import re

text = '好的，那我来推断一下最合理的主题——就以"AI工具如何改变日常工作流"为主题，采用第一人称个人经验分享的风格来写。开始创作！'
_llm_topic_re = re.compile(r'(?:就以|以|围绕|关于)(.{2,40}?)(?:为主题|为题|写|创作)', re.DOTALL)
_m = _llm_topic_re.search(text)
print('Match:', _m)
if _m:
    g = _m.group(1).strip()
    g = g.strip('\'"“”「」『』')
    print('Group 1 raw:', repr(_m.group(1)))
    print('Group 1 cleaned:', repr(g))
