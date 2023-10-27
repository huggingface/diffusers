# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import tempfile
import unittest
import uuid

import numpy as np
import torch
from huggingface_hub import delete_repo, hf_hub_download
from test_utils import TOKEN, is_staging_test
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.constants import WORKFLOW_NAME
from diffusers.utils.testing_utils import torch_device


class WorkflowFastTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_workflow_with_stable_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, return_workflow=True)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            output.workflow.save_pretrained(tmpdirname)

            components = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**components)
            sd_pipe = sd_pipe.to(torch_device)
            sd_pipe.set_progress_bar_config(disable=None)
            sd_pipe.load_workflow(tmpdirname)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        workflow_image_slice = image[0, -3:, -3:, -1]

        self.assertTrue(np.allclose(image_slice, workflow_image_slice))


@is_staging_test
class WorkflowPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-workflow-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def get_pipeline_components(self):
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_vocab = {"<|startoftext|>":0,"<|endoftext|>":1,"!":2,"\"":3,"#":4,"$":5,"%":6,"&":7,"'":8,"(":9,")":10,"*":11,"+":12,",":13,"-":14,".":15,"/":16,"0":17,"1":18,"2":19,"3":20,"4":21,"5":22,"6":23,"7":24,"8":25,"9":26,":":27,";":28,"<":29,"=":30,">":31,"?":32,"@":33,"A":34,"B":35,"C":36,"D":37,"E":38,"F":39,"G":40,"H":41,"I":42,"J":43,"K":44,"L":45,"M":46,"N":47,"O":48,"P":49,"Q":50,"R":51,"S":52,"T":53,"U":54,"V":55,"W":56,"X":57,"Y":58,"Z":59,"[":60,"\\":61,"]":62,"^":63,"_":64,"`":65,"a":66,"b":67,"c":68,"d":69,"e":70,"f":71,"g":72,"h":73,"i":74,"j":75,"k":76,"l":77,"m":78,"n":79,"o":80,"p":81,"q":82,"r":83,"s":84,"t":85,"u":86,"v":87,"w":88,"x":89,"y":90,"z":91,"|":92,"}":93,"~":94,"¡":95,"¢":96,"£":97,"¤":98,"¥":99,"¦":100,"§":101,"¨":102,"©":103,"ª":104,"«":105,"¬":106,"®":107,"¯":108,"°":109,"±":110,"²":111,"³":112,"´":113,"µ":114,"¶":115,"·":116,"¸":117,"¹":118,"º":119,"»":120,"¼":121,"½":122,"¾":123,"¿":124,"Â":125,"Ã":126,"Ä":127,"Å":128,"Æ":129,"Ç":130,"È":131,"É":132,"Ê":133,"Ë":134,"Ì":135,"Í":136,"Î":137,"Ï":138,"Ð":139,"Ñ":140,"Ö":141,"×":142,"Ø":143,"Ù":144,"Ü":145,"à":146,"á":147,"â":148,"ã":149,"ä":150,"å":151,"æ":152,"ç":153,"è":154,"é":155,"ë":156,"ì":157,"ï":158,"Ċ":159,"Ġ":160,"Ģ":161,"ģ":162,"Ĥ":163,"ĥ":164,"Ħ":165,"ħ":166,"Ĩ":167,"ĩ":168,"Ī":169,"ī":170,"Ĭ":171,"ĭ":172,"Į":173,"į":174,"İ":175,"ı":176,"Ĳ":177,"ĳ":178,"Ĵ":179,"ĵ":180,"Ķ":181,"ķ":182,"ĸ":183,"Ĺ":184,"ĺ":185,"Ļ":186,"ļ":187,"Ľ":188,"ľ":189,"Ŀ":190,"ŀ":191,"Ł":192,"ł":193,"Ń":194,"e</w>":195,"d</w>":196,"a</w>":197,"o</w>":198,"n</w>":199,"±</w>":200,"l</w>":201,"m</w>":202,"h</w>":203,"r</w>":204,"i</w>":205,"s</w>":206,"Z</w>":207,"t</w>":208,"f</w>":209,"k</w>":210,"y</w>":211,"b</w>":212,"F</w>":213,"g</w>":214,"7</w>":215,"0</w>":216,"p</w>":217,"L</w>":218,"H</w>":219,"¡</w>":220,"Ī</w>":221,"1</w>":222,"Ģ</w>":223,"c</w>":224,"ĩ</w>":225,"6</w>":226,"A</w>":227,"z</w>":228,"u</w>":229,"S</w>":230,"2</w>":231,"v</w>":232,"4</w>":233,"M</w>":234,"T</w>":235,"8</w>":236,"I</w>":237,"N</w>":238,"C</w>":239,"5</w>":240,"¹</w>":241,"9</w>":242,"3</w>":243,"ī</w>":244,"P</w>":245,"E</w>":246,"»</w>":247,"V</w>":248,"İ</w>":249,"w</w>":250,"J</w>":251,"ł</w>":252,".</w>":253,"K</w>":254,"D</w>":255,"Ķ</w>":256,"¸</w>":257,"B</w>":258,"©</w>":259,"º</w>":260,"µ</w>":261,"Ĥ</w>":262,"X</w>":263,"R</w>":264,"O</w>":265,"«</w>":266,"Ļ</w>":267,"U</w>":268,"x</w>":269,"[</w>":270,"¿</w>":271,"³</w>":272,"ģ</w>":273,"W</w>":274,"§</w>":275,"-</w>":276,"ĸ</w>":277,"Ħ</w>":278,",</w>":279,"q</w>":280,"ħ</w>":281,"¨</w>":282,"G</w>":283,"²</w>":284,"ĺ</w>":285,"ª</w>":286,"¯</w>":287,"j</w>":288,"]</w>":289,"ļ</w>":290,"Ŀ</w>":291,"¤</w>":292,"ŀ</w>":293,"½</w>":294,"Ĳ</w>":295,"'</w>":296,"Ń</w>":297,"°</w>":298,"ľ</w>":299,"></w>":300,"¶</w>":301,"į</w>":302,"¦</w>":303,"|</w>":304,"¼</w>":305,"¢</w>":306,"´</w>":307,"Ĩ</w>":308,"Q</w>":309,"Y</w>":310,"Ľ</w>":311,"ĵ</w>":312,"ĳ</w>":313,"ķ</w>":314,"Ĭ</w>":315,"¾</w>":316,";</w>":317,"(</w>":318,"¬</w>":319,"@</w>":320,"ĭ</w>":321,"Ĺ</w>":322,"£</w>":323,"Į</w>":324,"#</w>":325,"·</w>":326,"*</w>":327,"Ĵ</w>":328,"®</w>":329,")</w>":330,"^</w>":331,"ı</w>":332,"Ġ</w>":333,"_</w>":334,"Ł</w>":335,"}</w>":336,"ĥ</w>":337,"\\</w>":338,"¥</w>":339,"<</w>":340,"+</w>":341,"=</w>":342,"~</w>":343,"\"</w>":344,"!</w>":345,"?</w>":346,"`</w>":347,"$</w>":348,"Ċ</w>":349,"/</w>":350,"%</w>":351,"&</w>":352,":</w>":353,"Ġt":354,"Ġth":355,"Ġa":356,"Ġthe</w>":357,"in":358,"Ġo":359,"Ġ,</w>":360,"Ġs":361,"ed</w>":362,"Ġw":363,"er":364,"Ġ.</w>":365,"Ġi":366,"re":367,"Ġc":368,"nd</w>":369,"Ġf":370,"Ġb":371,"at":372,"Ġof</w>":373,"er</w>":374,"en":375,"ar":376,"or":377,"it":378,"Ġp":379,"Ġh":380,"Ġand</w>":381,"on":382,"ing</w>":383,"an":384,"ro":385,"Ġm":386,"Ġd":387,"es</w>":388,"Ġin</w>":389,"on</w>":390,"Ġto</w>":391,"ou":392,"is":393,"Ġa</w>":394,"ic":395,"ĠT":396,"al":397,"Ġl":398,"Ġ=</w>":399,"Ġre":400,"Ġ\"</w>":401,"es":402,"ĠS":403,"as</w>":404,"al</w>":405,"il":406,"el":407,"ion</w>":408,"ĠA":409,"ĠC":410,"Ġ1":411,"ĠĊ</w>":412,"ur":413,"ĠTh":414,"Ġn":415,"as":416,"Ġ@":417,"ec":418,"om":419,"ac":420,"Ġe":421,"Ġwas</w>":422,"ĠM":423,"or</w>":424,"an</w>":425,"am":426,"en</w>":427,"ol":428,"Ġin":429,"Ġg":430,"Ġ'</w>":431,"ĠB":432,"ly</w>":433,"at</w>":434,"iv":435,"ts</w>":436,"ĠThe</w>":437,"us":438,"-@</w>":439,"Ġ@-@</w>":440,"is</w>":441,"ĠI":442,"Ġwh":443,"ig":444,"ĠH":445,"Ġst":446,"os":447,"un":448,"th":449,"ĠP":450,"Ġwit":451,"Ġthat</w>":452,"ir":453,"Ġas</w>":454,"em":455,"Ġon</w>":456,"ra":457,"Ġfor</w>":458,"ĠR":459,"et":460,"ow":461,"Ġ2":462,"id":463,"ĠD":464,"le</w>":465,"Ġwith</w>":466,"la":467,"ent</w>":468,"im":469,"ĠF":470,"ea":471,"ion":472,"Ġby</w>":473,"Ġ)</w>":474,"Ġ(</w>":475,"Ġal":476,"Ġcon":477,"ent":478,"ĠW":479,"Ġis</w>":480,"ere</w>":481,"ĠG":482,"ĠN":483,"ĠL":484,"Ġha":485,"ers</w>":486,"ri":487,"th</w>":488,"ted</w>":489,"uc":490,"ĠJ":491,"Ġ19":492,"ev":493,"ul":494,"Ġv":495,"ce</w>":496,"ation</w>":497,"rom</w>":498,"Ġbe":499,"ĠE":500,"in</w>":501,"Ġthe":502,"Ġfrom</w>":503,"ĠO":504,"ter</w>":505,"Ġpro":506,"Ġar":507,"ad":508,"Ġcom":509,"ic</w>":510,"ag":511,"Ġhis</w>":512,"Ġsh":513,"Ġat</w>":514,"ov":515,"ies</w>":516,"oo":517,"pp":518,"st":519,"ch":520,"Ġr":521,"Ġ20":522,"ay</w>":523,"if":524,"Ġwere</w>":525,"Ġch":526,"ut</w>":527,"st</w>":528,"ut":529,"ds</w>":530,"op":531,"um":532,"Ġit</w>":533,"oc":534,"ter":535,"le":536,"igh":537,"ud":538,"Ġex":539,"ions</w>":540,"ate</w>":541,"ity</w>":542,"ated</w>":543,"Ġun":544,"ep":545,"qu":546,"Ġno":547,"ĠK":548,"ive</w>":549,"ist":550,"Ġon":551,"ame</w>":552,"oun":553,"ir</w>":554,"ab":555,"Ġâ":556,"ing":557,"Ġhe</w>":558,"ld</w>":559,"ug":560,"ich</w>":561,"Ġan</w>":562,"ed":563,"Ġk":564,"ĠâĢ":565,"Ġhad</w>":566,"ve</w>":567,"ain":568,"Ġse":569,"tion</w>":570,"ore</w>":571,"res":572,"Ġwhich</w>":573,"ĠIn</w>":574,"od":575,"ther</w>":576,"ak":577,"Ġsp":578,"ar</w>":579,"Ġy":580,"ĠCh":581,"ong</w>":582,"Ġac":583,"est</w>":584,"ĠU":585,"ap":586,"ff":587,"ally</w>":588,"rit":589,"ĠSt":590,"ub":591,"ge</w>":592,"ber</w>":593,"et</w>":594,"Ġbe</w>":595,"ear":596,"Ġrec":597,"ers":598,"Ġfir":599,"ot":600,"Ġare</w>":601,"Ġan":602,"ch</w>":603,"og":604,"ia</w>":605,"est":606,"ine</w>":607,"ill":608,"and":609,"el</w>":610,"ary</w>":611,"ew</w>":612,"id</w>":613,"Ġfor":614,"Ġ;</w>":615,"Ġcomp":616,"ĠV":617,"Ġinc":618,"tr":619,"Ġ200":620,"Ġtheir</w>":621,"us</w>":622,"Ġbut</w>":623,"ran":624,"ical</w>":625,"Ġfirst</w>":626,"Ġde":627,"Ġint":628,"Ġro":629,"so</w>":630,"ĠâĢĵ</w>":631,"Ġnot</w>":632,"ding</w>":633,"fter</w>":634,"ure</w>":635,"Ġpar":636,"Ġ:</w>":637,"ian</w>":638,"Ġtw":639,"ould</w>":640,"Ġalso</w>":641,"Ġits</w>":642,"Ġwor":643,"um</w>":644,"Ġor</w>":645,"ost</w>":646,"00</w>":647,"our":648,"ard</w>":649,"Ġres":650,"mp":651,"ue</w>":652,"Ġab":653,"ish</w>":654,"Ġcont":655,"Ġad":656,"own</w>":657,"all</w>":658,"oug":659,"Ġher</w>":660,"ast</w>":661,"Ġen":662,"ome</w>":663,"all":664,"ded</w>":665,"ow</w>":666,"Ġhave</w>":667,"Ġus":668,"ear</w>":669,"ack</w>":670,"duc":671,"ial</w>":672,"ss":673,"ents</w>":674,"ain</w>":675,"ting</w>":676,"Ġone</w>":677,"ess":678,"Ġhas</w>":679,"ight</w>":680,"av":681,"Ġev":682,"out</w>":683,"ay":684,"ence</w>":685,"Ġbeen</w>":686,"ew":687,"Ġtwo</w>":688,"Ġcl":689,"der</w>":690,"ime</w>":691,"ks</w>":692,"ess</w>":693,"ish":694,".@</w>":695,"Ġ@.@</w>":696,"Ġpla":697,"Ġpl":698,"Ġor":699,"up</w>":700,"ment</w>":701,"uring</w>":702,"oll":703,"ĠIn":704,"Ġthis</w>":705,"Ġbec":706,"Ġcomm":707,"Ġdis":708,"ater</w>":709,"age</w>":710,"Ġapp":711,"ous</w>":712,"ey</w>":713,"il</w>":714,"per":715,"ĠAl":716,"ional</w>":717,"lud":718,"ely</w>":719,"tt":720,"ile</w>":721,"iz":722,"Ġj":723,"Ġwho</w>":724,"Ġag":725,"ib":726,"Ġthey</w>":727,"for":728,"Ġov":729,"ath":730,"eg":731,"Ġsc":732,"ip":733,"Ġ201":734,"Ġ3":735,"Ġper":736,"ory</w>":737,"Ġdes":738,"ide</w>":739,"Ġser":740,"se</w>":741,"ĠHe</w>":742,"land</w>":743,"ations</w>":744,"ric":745,"it</w>":746,"res</w>":747,"ered</w>":748,"Ġpre":749,"ĠSh":750,"ance</w>":751,"ort</w>":752,"ant</w>":753,",@</w>":754,"Ġ@,@</w>":755,"ell</w>":756,"ĠY":757,"ned</w>":758,"ell":759,"ite</w>":760,"Ġinclud":761,"Ġrep":762,"Ġafter</w>":763,"Ġsuc":764,"ree</w>":765,"any</w>":766,"im</w>":767,"ort":768,"Ġ18":769,"Ġsu":770,"ade</w>":771,"our</w>":772,"ĠUn":773,"ĠIt</w>":774,"ik":775,"ĠMar":776,"ember</w>":777,"Ġ1</w>":778,"een</w>":779,"and</w>":780,"Ġsec":781,"ice</w>":782,"Ġtime</w>":783,"ĠAn":784,"Ġinto</w>":785,"Ġfin":786,"Ġother</w>":787,"Ġatt":788,"ill</w>":789,"ren":790,"ach":791,"ass":792,"eral</w>":793,"ese</w>":794,"sh":795,"als</w>":796,"ition</w>":797,"ough</w>":798,"les</w>":799,"amp":800,"Ġwould</w>":801,"Ġmore</w>":802,"roug":803,"rib":804,"ery</w>":805,"ace</w>":806,"ĠA</w>":807,"Ġplay":808,"ited</w>":809,"ked</w>":810,"ist</w>":811,"ied</w>":812,"Ġ2</w>":813,"ased</w>":814,"ings</w>":815,"ang":816,"am</w>":817,"ip</w>":818,"Ġbo":819,"able</w>":820,"ty</w>":821,"Ġchar":822,"Ġcent":823,"etw":824,"ates</w>":825,"rop":826,"ĠI</w>":827,"und</w>":828,"ĠAm":829,"ces</w>":830,"oin":831,"Ġinter":832,"up":833,"ct":834,"one</w>":835,"Ġtra":836,"ant":837,"ect":838,"Ġall</w>":839,"ef":840,"Ġcons":841,"ubl":842,"ning</w>":843,"ans</w>":844,"Ġfe":845,"ust</w>":846,"Ġ0":847,"Ġrem":848,"ase</w>":849,"ong":850,"Ġwhen</w>":851,"eb":852,"ĠWh":853,"Ġear":854,"ever</w>":855,"Ġover</w>":856,"Ġkn":857,"aus":858,"Ġpos":859,"ad</w>":860,"erm":861,"Ġshe</w>":862,"Ġra":863,"Ġduring</w>":864,"ason</w>":865,"vi":866,"Ġexp":867,"Ġlea":868,"Ġel":869,"Ġ4":870,"Ġonly</w>":871,"ond</w>":872,"Ġdec":873,"Ġacc":874,"Ġoff":875,"iss":876,"Ġfl":877,"ĠEn":878,"ot</w>":879,"ens":880,"ose</w>":881,"ake</w>":882,"om</w>":883,"Ġsev":884,"ach</w>":885,"etween</w>":886,"ern":887,"Ġ3</w>":888,"Ġpr":889,"Ġgro":890,"ruc":891,"Ġdi":892,"Ġ199":893,"ĠAr":894,"Ġgame</w>":895,"Ġhim</w>":896,"ook</w>":897,"Ġup</w>":898,"Ġabout</w>":899,"Ġrel":900,"form":901,"Ġthree</w>":902,"att":903,"ĠCom":904,"Ġsa":905,"ears</w>":906,"Ġ5":907,"ry</w>":908,"Ġimp":909,"Ġmost</w>":910,"fer":911,"Ġpres":912,"Ġfil":913,"Ġbetween</w>":914,"Ġbeg":915,"ph":916,"ors</w>":917,"Ġthan</w>":918,"Ġrecor":919,"ob":920,"eric":921,"ating</w>":922,"Ġthroug":923,"king</w>":924,"Ġout</w>":925,"Ġnum":926,"ood</w>":927,"ollow":928,"act":929,"uil":930,"Ġcre":931,"olog":932,"ational</w>":933,"Ġproduc":934,"Ġwhile</w>":935,"Ġlater</w>":936,"Ġwrit":937,"ex":938,"Ġstar":939,"Ġspec":940,"ee":941,"ished</w>":942,"Ġreg":943,"ision</w>":944,"outh</w>":945,"Ġrele":946,"Ġass":947,"Ġseason</w>":948,"Ġmade</w>":949,"ily</w>":950,"ru":951,"oy":952,"tur":953,"te</w>":954,"Ġqu":955,"Ġmov":956,"ury</w>":957,"ĠAmeric":958,"ement</w>":959,"cc":960,"ound</w>":961,"Ġlar":962,"Ġform":963,"ect</w>":964,"Ġdef":965,"Ġmus":966,"ĠPar":967,"Ġme":968,"Ġsub":969,"way</w>":970,"op</w>":971,"oh":972,"eld</w>":973,"ie</w>":974,"emp":975,"ames</w>":976,"ern</w>":977,"Ġnor":978,"ived</w>":979,"evel":980,"Ġsuch</w>":981,"ards</w>":982,"Ġind":983,"ike</w>":984,"Ġgen":985,"ert":986,"Ġyear</w>":987,"Ġused</w>":988,"Ġnew</w>":989,"Ġ5</w>":990,"Ġalb":991,"sp":992,"yp":993,"Ġwith":994,"Ġwhere</w>":995,"ics</w>":996,"ĠThis</w>":997,"Ġthem</w>":998,"wn</w>":999}
            vocab_path = os.path.join(tmpdir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(dummy_vocab, f)

            merges = "Ġ t\nĠt h"
            merges_path = os.path.join(tmpdir, "merges.txt")
            with open(merges_path, "w") as f:
                f.writelines(merges)
            tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=merges_path)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_push_to_hub(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        inputs = self.get_dummy_inputs()
        workflow = pipeline(**inputs, return_workflow=True).workflow
        workflow.push_to_hub(self.repo_id, token=TOKEN)

        local_path = hf_hub_download(repo_id=self.repo_id, filename=WORKFLOW_NAME, token=TOKEN)
        with open(local_path) as f:
            locally_loaded_workflow = json.load(f)

        for k in workflow:
            assert workflow[k] == locally_loaded_workflow[k]

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

    def test_push_to_hub_in_organization(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        inputs = self.get_dummy_inputs(device="cpu")
        workflow = pipeline(**inputs, return_workflow=True).workflow
        workflow.push_to_hub(self.org_repo_id, token=TOKEN)

        local_path = hf_hub_download(repo_id=self.org_repo_id, filename=WORKFLOW_NAME, token=TOKEN)
        with open(local_path) as f:
            locally_loaded_workflow = json.load(f)

        for k in workflow:
            assert workflow[k] == locally_loaded_workflow[k]

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)
