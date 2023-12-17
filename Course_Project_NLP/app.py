from gpt import Chatbot
import gradio as gr
from give_star import star_evaluator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--text", default=None, type=str, help="The path of text prompt.")
parser.add_argument("--modelpath", default=None, type=str, help="The path of model parameters for initialization.")
parser.add_argument("--model", choices=["uie-senta-base", "uie-senta-medium", "uie-senta-mini", "uie-senta-micro", "uie-senta-nano"], default="uie-senta-base", type=str, help="Select the pretrained model for few-shot learning.")
args = parser.parse_args()
args.modelpath="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/model_6"
args.text="位置在宇宙中心五道口地铁站附近。位于五道口国际美食苑的2楼。和其它分店不一样，一进门特别大一个厅，感觉像个大食堂，环境有点吵闹。饭点是一定要等位的，好在叫号非常快。分析第一是因为日昌本身人气就旺，第二因为他家菜量普遍很大，性价比还比较高。\\r\\n【豉汁排骨饭】推荐，排骨很多很入味。汤汁不够浓还可以要。服务员会给你拌好，最爱吃底下的锅巴，很香。两个人吃一份就够了。尝试过其他几种煲仔饭还是觉得排骨的最好吃~\\r\\n【腊味煲仔饭】很大一碗，腊肉很多，就是有点油。\\r\\n【蟹黄豆腐】蟹粉吃起来沙沙的，味道浓郁，豆腐很嫩。放在纸炉上加热，味道不错，推荐。\\r\\n【杨枝甘露】本身就很爱，别家都是一小碗，这的杨枝甘露是成扎的，喝起来很过瘾。清凉爽口！非常好喝！每次都眼大肚子小的点个大扎，最后喝不了......\\r\\n【纸包鸡翅】必点菜品，鸡翅烤出来焦香焦香的，肉很入味很嫩，因为被锡纸包住烤的，吃的时候小心被烫到。味道有点甜又有点辣，很好吃。两个人点一份合适，吃多了会有点腻。\\r\\n总体来说是个很适合聚餐的地方，饭菜味道普遍都还不错。但去过几次发现，水平略有下降，这点希望改进。还有作为粤菜馆他家的粥非常一般......重点提醒：日昌有很多假冒的，吃过一次，味道差太多，一定要认准。"

path="/mnt/ve_share/liushuai/PaddleNLP/chatglm-6b"

chatlog="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/chat_logs/logs.txt"
requirements="生成一段关于餐厅的评价，要包括口味、外观、分量、推荐程度、价格水平、性价比、折扣力度、位于商图附近、交通方便、是否容易寻找、排队时间、服务人员态度、停车方便、点菜/上菜速度、装修、嘈杂情况、就餐空间、卫生情况其中的一个或多个方面"
suggestionslog="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/chat_logs/seuugestions.txt"
purelog="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/chat_logs/raw_data_gpt.txt"
with open(chatlog,"w+")as f:
    pass
with open(purelog,"w+")as f:
    pass
def writelog(text):
    with open(chatlog,"a+")as f:
        f.write(text+"\n")
def botactivation(): #启动bot
    global bot
    bot=Chatbot(path)
    return "success!"
def chatbot(input_text): #生成一段
    writelog("You: " + input_text)
    response,_ =  bot.model.chat(bot.tokenizer, input_text)
    writelog("Bot: " + response.replace("\n", ""))
    with open(purelog,"a+")as k: #纯文本单独存一个文件 用于训练的raw_data
        k.write(response.replace("\n", "")+"\n")
    return response
def Super_agent(input_text):
    prompt="假设你是餐厅老板，你看到了用户的评论如前文所述，你将对整个餐厅做出怎样的改进？"
    response,_ =  bot.model.chat(bot.tokenizer, input_text+prompt)
    with open(suggestionslog,"a+")as f:
         f.write((input_text+prompt).replace("\n", ""))
         f.write(response.replace("\n", ""))
         f.write("\n")
    return response
def readlog():
    with open(chatlog,"r")as f: 
        texts=f.readlines()
    return "\n".join(texts[-10:])
star_evaluator=star_evaluator()

with gr.Blocks() as demo:
    with gr.Tab(label="chat"):
        out = gr.Textbox(id="output-textbox")
        with gr.Row():
                inp = gr.Textbox(id="chatbots",value=requirements)
                btn = gr.Button("Generate", id="generate-button")
                btn.click(fn=chatbot, inputs=inp, outputs=out)
                clear = gr.ClearButton([inp])
        with gr.Row():
                inp2 = gr.Textbox(label="superAgent",value=args.text)
                btn2 = gr.Button("Give Suggestions", id="generate-button")
                btn2.click(fn=Super_agent,inputs=inp2, outputs=out)
                clear = gr.ClearButton([inp2])
        activationbutton=gr.Button("start robot!")
        activatestatus = gr.Textbox("activate your chatbot first!")
        activationbutton.click(botactivation,outputs=activatestatus)
#     with gr.Tab(label="chat"): #TODO chatbot一直调不通，换成两个tab来解决
#         chatbot_page = gr.Chatbot()
#         inp = gr.Textbox()
#         clear = gr.ClearButton([inp,chatbot_page])
#         inp.submit(chatbot, [inp, chatbot], [inp, chatbot])
    with gr.Tab(label="history"):
        history_out = gr.Textbox()
        btn = gr.Button("View history")
        btn.click(fn=readlog, inputs=None, outputs=history_out)
    with gr.Tab(label="rating system"):
        textinp=gr.Textbox(value=args.text)
        options=gr.Dropdown(["uie-senta-base-finetune", "uie-senta-medium-finetune", "uie-senta-mini-finetune","uie-senta-micro-finetune",
                             "uie-senta-base", "uie-senta-medium", "uie-senta-mini","uie-senta-micro",
                             ],value="uie-senta-medium-finetune", label="model", info="Which model do you want?")
        with gr.Row():
                output1=gr.Textbox(label="calculate results")
                gradingbutton=gr.Button("grading")
                def get_first_output(input1,input2):
                    return star_evaluator.comment2stars(input1,input2)[0]
                gradingbutton.click(fn=get_first_output,inputs=[textinp,options],outputs=output1)
        with gr.Row():
                output2=gr.Textbox(label="linear results")
                gradingbutton2=gr.Button("grading with linear")
                gradingbutton2.click(fn=star_evaluator.comment2stars_linear,inputs=[textinp,options],outputs=output2)
        with gr.Row():
                colors = ["green", "red", "gray", "blue"]
                t1=gr.Radio(["正向", "中性","负向", "未提及"],label="口味")
                t2=gr.Radio(["正向", "中性","负向", "未提及"],label="外观")
                t3=gr.Radio(["正向", "中性","负向", "未提及"],label="分量")
                t4=gr.Radio(["正向", "中性","负向", "未提及"],label="推荐程度")
                t5=gr.Radio(["正向", "中性","负向", "未提及"],label="价格水平")
                t6=gr.Radio(["正向", "中性","负向", "未提及"],label="性价比")
                t7=gr.Radio(["正向", "中性","负向", "未提及"],label="折扣力度")
                t8=gr.Radio(["正向", "中性","负向", "未提及"],label="位于商图附近")
                t9=gr.Radio(["正向", "中性","负向", "未提及"],label="交通方便")
                t10=gr.Radio(["正向", "中性","负向", "未提及"],label="是否容易寻找")
                t11=gr.Radio(["正向", "中性","负向", "未提及"],label="排队时间")
                t12=gr.Radio(["正向", "中性","负向", "未提及"],label="服务人员态度")
                t13=gr.Radio(["正向", "中性","负向", "未提及"],label="停车方便")
                t14=gr.Radio(["正向", "中性","负向", "未提及"],label="点菜、上菜速度")
                t15=gr.Radio(["正向", "中性","负向", "未提及"],label="装修")
                t16=gr.Radio(["正向", "中性","负向", "未提及"],label="嘈杂情况")
                t17=gr.Radio(["正向", "中性","负向", "未提及"],label="就餐空间")
                t18=gr.Radio(["正向", "中性","负向", "未提及"],label="卫生情况")
                gradingbutton2=gr.Button("show_aspects")
                def get_other_output(input1,input2):
                    return star_evaluator.comment2stars(input1,input2)[1:]
                gradingbutton2.click(fn=get_other_output,inputs=[textinp,options],outputs=[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18])
demo.launch()
