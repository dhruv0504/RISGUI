# app/services/farfield_core.py
# Far-Field compute & plotting service (wrapped from your provided script)

# Standard I/O utilities (not heavily used here but available)
import io
import sys

# Used to capture full exception stack traces
import traceback

# Numerical computing library (arrays, math, vectorization)
import numpy as np

# Plotly for interactive 2D/3D visualization
import plotly.graph_objects as go

# PIL image module (not directly used in this snippet but imported)
from PIL import Image

# Functional programming utilities
import functools

# LRU cache decorator for memoizing compute_fields results
from functools import lru_cache

# Small helpers

# Wraps angle(s) in radians into range [-π, π]
# Ensures phase continuity and prevents overflow
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# sample random vector from your paste
# Predefined random phase sample used for per-element phase randomization
phi_ele_rand_sample = np.array([0.219827130698740, 2.98461967174048, 0.497030697989493, 0.899896229536556, 2.15867977892526, 0.443435248995979, 1.60876765916016, 2.26611537562385, 2.91805305438595, 2.29997207687589, 2.35571703325974, 1.27963227277936, 0.752385351454657, 1.63630002613963, 0.688251825085977, 2.64643965036121, 2.08266515689846, 2.56427877547971, 2.49404003633173, 1.47373746123896, 0.972401401180031, 2.16009391274997, 3.10028769518068, 2.41881860813769, 2.60620509363067, 2.21823071195037, 1.87030451923823, 2.36522380472919, 1.56049882641230, 2.71789439022493, 0.213715304117615, 3.04277681043736, 0.310252104183716,
1.71837919648984, 1.26596730566785, 0.336276565844324, 2.27503512116018, 1.92793995599068, 2.45976712160923, 1.78009105917144, 2.54883430871032, 1.81199480530103, 2.96575518014727, 2.73774766907021, 1.59467936742524, 2.47816106018460, 1.48606958637337, 2.60375732725403, 1.01310578797340, 3.06665478389739, 0.874025760238266, 0.228804721353759, 2.36003856141073, 2.61125606401758, 2.89761057813510, 1.02737711759663, 2.52605808826951, 1.69096329288932, 1.45548378856132, 2.57846295851701, 2.99050248675540, 0.239618349132054, 2.22635580906413, 0.738043190092256, 1.25316735367965, 0.842337729817777,
2.61541771943221, 3.12705905015932, 2.04125291408473, 2.21153304248241, 2.92891687765692, 2.16032419323750, 1.78553649553246, 1.19646780015380, 1.99359024759922, 1.14111636760235, 1.28057425370241, 1.15830434296629, 1.47151810459291, 1.58152123177630, 2.86053252481442, 0.648522182420109, 1.06375481872993, 1.80366883627629, 1.52974337450673, 0.823785239076857, 1.82084656528219, 2.75934784654565, 0.191480770877785, 1.38505427593173, 0.264704318007911, 1.76946315484690, 1.69429415704724, 2.41292505702392, 0.732274686763429, 1.84525133519519, 1.44190892819883, 2.70485322377502, 2.07607781954558,
1.11174418850480, 1.09071792289046, 0.797078926313911, 2.99245542384257, 0.936826038924080, 0.497647370232264, 1.13504817917745, 2.32989650988914, 2.21765028567742, 2.20191770125035, 0.0195588879947849, 1.17604139087184, 2.83213201481320, 1.00010977319580, 1.87579132515556, 0.935551157529470, 0.392744261840316, 1.22005531563296, 2.56884180939871, 3.08245438433163, 2.70802093400720, 0.263331067410229, 1.06095350787230, 0.741821654373007, 0.998415071998961, 3.09273587847640, 1.72238097229522, 2.35384225208077, 2.64475522522102, 0.524299477430201, 2.83716481481866, 0.330257187681317, 2.34077899801865,
2.29138952164611, 2.25399723116366, 0.419188261262462, 1.40048744593145, 1.59840222782399, 1.66658440043671, 2.70088074829411, 2.12913560067177, 2.53161555381118, 1.66894869596304, 3.00303534418426, 0.209471799317047, 1.70122905682320, 0.884862065930330, 1.51079296383476, 2.15156302419513, 0.654263024292310, 1.91059409396702, 1.02471266691487, 2.76726309987498, 0.419071537363133, 0.321723560380429, 3.01315461681935, 0.480355798897516, 0.479212514962152, 0.488682867885581, 0.281390578221451, 1.42761686371483, 2.10139779455066, 2.61161125693274, 2.48259690073665, 2.23904738346855, 1.48470884994412,
2.22609533398725, 3.00983262727531, 1.58894079088047, 0.958352626036071, 2.48126501275532, 0.742631548211931, 0.736084059725873, 1.45989621679682, 1.94586102443051, 1.93311432310414, 0.385234348283003, 0.388908839154729, 0.893654277302504, 2.31137278092993, 1.29216251469346, 2.60432342410523, 2.93774763724997, 1.25370488157450, 0.164026888252000, 1.79443406262497, 2.34887450291146, 1.00607480949966, 1.54859883414861, 0.696342665109585, 2.95081548963881, 1.51520675383460, 1.69644773949043, 0.694471360452799, 0.301420112869988, 0.189015230349268, 2.57456306107027, 2.42367057453123, 0.614796270910489,
2.81209661843906, 2.14979231222819, 2.06354408002382, 3.11137430684799, 0.105845889755238, 1.33283095762596, 1.53932910686707, 1.83313335775680, 0.261599627267670, 2.07393693577149, 0.164321697316658, 1.74933471532356, 2.23689329185278, 1.53281643592544, 1.94024965203053, 0.671602058374640, 2.02839650945857, 1.19582302594537, 0.325823496814061, 1.18598741399057, 0.825801581785865, 0.758021108127581, 1.95697362040097, 1.64282601595231, 1.29822466857210, 0.684213858818427, 2.69722573625974, 2.70493720324631, 0.892020712799620, 1.93331365331770, 2.44883849254340, 2.99974129885523, 2.88901880256175,
1.20893418726973, 0.510958105499636, 2.50306840710463, 0.357567215390901, 0.498961249930249, 1.11786699595932, 2.66331207792344, 1.83086112711087, 1.84153144681656, 2.90860582449915, 1.80666166106915, 0.0313449107089498, 2.54273373510786, 1.91262786993564, 1.50769329262719, 0.843329629215697, 0.810833703879609, 1.51113602438121, 0.714214335918747, 0.152688623104308, 0.531688841246168, 0.811933021776786, 0.621752782889969, 1.90284088076767, 2.58774132106069, 2.54662201161908, 2.52030191599744, 2.22460378122810, 2.69981515373615, 2.45384938358587, 0.640246690075069, 3.12067985647598, 0.294130130051291,
2.04397843782980, 0.675996993675282, 0.766149946098465, 1.06716071293672, 0.621602875353898, 1.59226401263341, 2.98689429393634, 1.23969513807563, 1.83615044862489, 1.90549295601230, 2.24511678118373, 1.26146365921374, 2.69765740622386, 2.89178885769794, 2.35882993926112, 0.897212802966457, 2.50327062549347, 0.448512083118626, 1.58509117218227, 1.91853278201134, 2.21104265340085, 1.20429791111958, 2.28922653117741, 2.78748703241838, 0.175452675813458, 0.434219446415085, 2.71139859673429, 1.32495565125464, 1.29218158904557, 3.01323138928473, 2.35696510079443, 3.08189332735102, 0.733611539450947,
0.302305286090706, 1.20820061176750, 1.57165257083367, 1.79151164831602, 3.06817864958901, 1.54835021596077, 1.25941066800711, 3.12585409692803, 0.819949425618981, 2.09017922351304, 3.02930150153351, 2.10848316533558, 0.939885531113136, 1.66858319476968, 0.00459670037259909, 2.77650443094303, 1.27040935534839, 0.946265257810930, 2.98634014420648, 1.44716897966606, 0.903671307512088, 0.265862426049558, 1.82897660562250, 0.480879495381778, 0.229632535281315, 1.82389803782478, 0.901683437602652, 1.13700547834130, 2.27710393324588, 2.69646566249046, 1.09300936530187, 3.02141292831401, 2.99572552709258,
0.647285509415166, 2.41351240541487, 1.93375043175735, 2.88688013579866, 1.89292663332986, 2.20583229984629, 2.33632377592788, 1.20983448362551, 0.790122456421988, 0.115484121044937, 1.48321775332722, 2.02655448804669, 0.876409915339451, 1.62690803003600, 0.771784507057351, 0.934649495155067, 2.04352463267912, 2.80039466008367, 2.70524574977282, 0.659467663034132, 1.25379050078840, 2.78936678357449, 0.805906517637067, 3.03729681775954, 1.94514019153597, 0.519449045435787, 2.59558079936504, 2.05992062976648, 1.71673142361406, 0.789574230452325, 0.126155002713480, 0.733170481670043, 1.13446974763878,
1.99005768437432, 3.09791853481002, 0.650801797671301, 2.37844934894241, 2.78448187788514, 1.48355385740138, 0.499241566066122, 2.54758278283724, 1.49699325531415, 0.365328057008699, 2.75115744372138, 1.99549686338127, 0.305618371637571, 2.85394672195353, 0.110006075934696, 0.124877347007208, 3.10567762465118, 2.15570306875150, 1.18340400059247, 1.58438136762006, 2.39859241709588, 0.153544702508209, 2.28056899299293, 2.20328707664792, 1.44164804717231, 1.82933309403728, 1.06528202862367, 0.536035546090639, 1.25410306505550, 2.88956527732006, 0.710142685240806, 1.13413598452498, 1.01964163490804,
0.262580954270873, 1.61058627412362, 2.61652201496655, 2.84192712134246, 2.27324360215424, 1.20321686797803, 0.936248193731221, 2.17307619316425, 2.76603701312890, 2.90455417110636, 0.255263122187734, 1.51636199666787, 0.402956556796092, 0.794543839882424, 2.77704706363477, 0.616622108620529, 0.381243750207925, 1.70806819725007, 0.988410231279342, 1.20021693214840, 2.48669393320323, 2.63636008796699, 2.13702563273201, 1.30979911525774, 2.01969789632887, 0.672556715428290, 1.93921342872880, 2.12117554355779, 1.88817096388184, 1.08796339364672, 1.14480066148856, 0.538723691203648, 2.49870205872176,
1.54775875234879, 1.11408055464335, 2.43492657192089, 0.743944242208389, 2.65412255692993, 2.56519513298333, 2.65850481180091, 1.16297629112912, 1.20395108098622, 2.70596291678723, 1.45741371229388, 1.79243009617396, 2.18437076220849, 3.01881064611566, 1.71629284218021, 1.99986428654048, 1.79351165015769, 2.91260871021763, 2.71359956856258, 0.533565742813057, 0.561399557235034, 0.764991532442059, 2.36178449898106, 0.625599089944829, 3.08799868985670, 2.22939573441124, 0.551148905937728, 2.69642078243447, 2.85700043421944, 3.02115366796370, 1.79259183849696, 1.76833725492724, 0.554997542841630,
1.61377036237889, 1.72307534178540, 0.519234399539646, 1.55161074151268, 1.68112047631243, 0.624571112809811, 1.95774287693431, 0.0826722473808538, 1.00151134189600, 1.67446742167096, 1.02658997307829, 1.89183706428442, 1.13704519767031, 0.423865680715214, 2.87082978949552, 2.01237493742719, 2.06960000581165, 2.12161280758943, 2.33909704512368, 2.64577887694223, 1.62312636249934, 0.477109596798208, 1.19589208758715, 2.57930852002200, 0.538357075230316, 1.03664791996849, 3.03626129397446, 2.53304289785290, 0.698023968548434, 3.14087989751437, 0.200241023703393, 1.33669463800697, 1.27026576817103,
1.25755718329301, 0.351615355865103, 1.33301160804235, 1.92751123762416, 3.10408607662610, 0.690838671320833, 1.11237851313816, 0.836423531735360, 0.915768082040860, 0.591843199623564, 0.0718156256460671, 1.41184487650189, 0.765418241687995, 2.72918493093326, 1.66067970360620, 2.87184032862271, 3.05969188935767, 1.83916989095434, 0.373772188165666, 2.91078797469824, 1.86472616228589, 2.77595935055116, 1.33353046673066, 1.90775534233144, 0.222310350852386, 2.90525827604751, 2.01715197677199, 0.328295388969595, 2.19982194736217, 1.24345597170613, 0.266736141296123, 0.673805234648707, 0.781624788727868,
0.712051831561837, 2.20855343026390, 2.36924022140608, 1.71935166835394, 1.73881944594920, 1.98100284996788, 3.09590431616614, 1.99264374282233, 1.88638694113359, 2.85629506510242, 1.79333617540434, 1.05375276045744, 3.00694134821264, 1.38205924909961, 1.88980179261020, 2.26277094376719, 2.13244505161093, 0.668402143125259, 0.256427395404666, 0.862302491375074, 2.72538938977712, 1.75727205366938, 1.45967119211323, 1.35183014872479, 2.43154848031807, 2.05435159152808, 2.06631744791853, 0.505862090917795, 1.35835186310871, 1.58677473204174, 1.17914113642465, 1.50913386274312, 1.07574856336506,
2.44147754710751, 1.20619521747471, 2.23542044365026, 1.51089685952846, 2.29078722896010, 2.94542921743268, 1.62500263109096, 2.83707414637207, 0.685472401585823, 2.74329946405973, 0.259787507457770, 1.46210795445408, 0.0688945702280719, 2.53928823803120, 0.563003251139228, 0.519595067677081, 0.570524118302906, 2.17221202772186, 0.671557901612046, 0.936524037462923, 2.41379605349773, 1.57440945568576, 2.85716795482173, 0.181750255329909, 1.37209664920906, 1.79779130466827, 1.77521083794083, 2.58809607550389, 0.396151150045565, 0.942844663519102, 0.00666653762674939, 2.98799188832328, 2.40740045029434,
2.36029163654173, 0.436256752203903, 1.09741970892142, 0.475450393049038, 1.56049403370123, 2.54045597705777, 1.98821595216094, 2.16267708787206, 2.00926909786890, 2.29123177727022, 2.70128495974171, 1.96963841409743, 0.567344530433829, 1.80109563421337, 0.513858235272062, 2.84644547625630, 0.242979525583014, 1.06353796987068, 1.82406382144816, 1.49299527950533, 2.52998892331149, 1.66748870837250, 0.714116467972470, 2.22891278590960, 0.466942937232651, 2.06753296013995, 1.99171491495565, 0.720386835027082, 0.572486194058969, 0.522611816211071, 0.470004539954372, 0.636949735282592, 3.00009187041832,
0.0499775203346953, 3.00812004820459, 0.0807146949730406, 3.05083412500419, 0.934924391844502, 1.64956572827352, 2.70911853168586, 2.81613905936723, 0.593794172330630, 2.07571170776185, 2.95696419975477, 3.06527433806207, 0.339088597943899, 0.562028500487626, 2.34536047900498, 0.155409946970315, 0.223948061488654, 1.53663313896857, 2.67002018513780, 3.13229807541168, 0.0137997994182283, 1.70465162287099, 2.70600510321325, 2.85614448036694, 2.65574711615935, 2.76106059781966, 2.34420063606476, 0.369103773707705, 1.59914058259123, 0.530400629347414, 2.61101379275392, 2.91543208998199, 0.532451036554243,
2.77634267235716, 1.21850722036690, 1.20187740573256, 0.852794511730255, 2.72653530570534, 2.32949867219251, 1.40703367871222, 2.22939622246284, 2.96670422092000, 0.547007432157162, 0.768421888797688, 2.01353822867190, 2.54033112069964, 2.68094408465968, 1.25072347483072, 0.362834501096907, 0.252210048313235, 1.13244242709893, 2.60408358526947, 0.674216276627760, 2.48512647027850, 2.05676399820759, 0.0821414891484141, 2.46858900714152, 2.89831728550305, 1.54664731553907, 2.62012488872977, 0.412659772896612, 2.38693017138596, 2.90828598436379, 2.61602799833061, 0.814932781266990, 0.669227619399456,
1.64090028099927, 1.24833414880369, 1.50516777313336, 3.12244037247612, 1.89902508221309, 2.96851806161525, 1.54076966008939, 1.37584969484548, 2.42736906997206, 2.33755406160623, 1.39142416473759, 0.166504575605426, 0.275900460429647, 2.50694682230782, 2.05957276671785, 0.101585137157028, 1.75007705968499, 2.26132327689182, 0.346857115500363, 0.680617805741627, 2.54789515129348, 0.435619152555019, 2.77056805022979, 2.90143693858459, 0.0400727824984256, 1.18488073859329, 0.527195969278519, 1.69715996641379, 0.319381958715320, 0.123363262010255, 2.93182572166390, 3.05234592149370, 1.13388881625622,
2.02383096387646, 0.213462745088759, 0.653174800532538, 0.124419154084547, 1.47453577031801, 0.471542761244834, 3.11428254104497, 1.34165593434667, 3.00138998013839, 2.27528915989090, 1.82492513609323, 1.69727027299692, 2.21620886495735, 0.0157987028345623, 2.45834582211003, 2.91181522665728, 0.0260615763214148, 2.59064634326267, 2.41065672583233, 3.13259794505077, 0.715193218745248, 2.88882703964628, 2.01690030183697, 0.330873113111868, 0.842452356496550, 2.39968595916793, 2.53058478714809, 0.327520451380898, 1.47579072056093, 0.688203227105200, 2.89877238060384, 1.00632322752048, 2.69405332979711,
0.816332382165120, 2.75851609019655, 0.591460161939569, 2.38507778206883, 0.0995548724299115, 2.01796797700490, 1.78087749506258, 1.18252546272700, 0.667740427380706, 2.48863410756649, 0.456924168672012, 1.53668632870619, 0.0403561040084870, 0.586258966221509, 1.52439643885142, 2.63336488460904, 0.443143536289710, 2.30032750848640, 2.17105074639249, 0.108361760918468, 1.53579038800635, 3.05171297631526, 0.353276427254192, 2.33487493307603, 2.00603688121308, 1.86668067427496, 1.56646882292669, 1.78396174273994, 1.33990262272311, 0.239513272012575, 0.912904895734672, 1.76348606087177, 1.98967577135206,
2.92411860092431, 3.07175240274402, 0.294042221333683, 2.07890164000706, 1.89366691280647, 1.48854271823448, 1.11921230838107, 1.49407373844061, 2.10807784303596, 3.01481430092844, 0.279866474065381, 2.50618185131215, 1.85597674399524, 2.86575062178678, 0.317707125279883, 0.921412295682781, 0.162069127046048, 1.58376538863375, 2.41392361736142, 0.889024778987588, 0.707990092598743, 1.04077710748294, 1.42393111581378, 2.31656216218034, 1.60185302698623, 1.20170377502047, 2.84465942199610, 3.03244665298391, 1.97376009926910, 0.414788153889359, 1.94245232449880, 1.20329314510687, 3.11392699564627,
0.901094490648649, 2.21856580722602, 1.68139891576104, 0.606990379433397, 2.16592424819031, 0.158509067291553, 0.579416862952421, 0.143439853611772, 2.78043988093487, 2.63829187294931, 0.371195650157908, 1.28935628930125, 0.377709255066085, 1.79728166510794, 2.98259707234157, 0.805457734507323, 3.10975397911384, 1.09895437191163, 0.655092357932704, 2.09175771455731, 3.05785258004480, 1.95626918473708, 0.199610176482807, 1.17341480567664, 0.522294602375483, 0.726581511113480, 0.164018258809929, 2.83295101991537, 2.49219937656376, 1.17185906261569, 2.61397694979576, 2.36824110500455, 1.95364053789622,
1.23808024178553, 1.12870500428870, 0.279138238392569, 1.07340996453489, 1.72369950756279, 1.44685252038596, 2.02774843462735, 1.61327330239127, 2.55859621052747, 0.305310045981052, 1.45679911549390, 1.85296629230104, 0.588018401765148, 1.92055029759576, 0.163180547079694, 1.80869955173588, 2.64630392705556, 1.56993544753682, 1.37923736181327, 0.468276637690176, 0.0888425151817762, 2.37714824712497, 2.50104150231505, 0.922232098842016, 0.361932666934231, 1.17838511233487, 2.60404650007688, 2.64451912583300, 2.08990823229943, 3.01636869466214, 2.96289224163824, 0.354055897326592, 2.03665508790157,
1.51049047657233, 0.208980917139073, 2.82043148673564, 1.56209452021196, 2.42312118579089, 0.189633536512643, 0.824534443622298, 2.04539474569305, 0.419729063559879, 2.00605059578983, 1.20933439920119, 2.40551160648711, 2.05119511047528, 1.19848324491639, 0.942536175477781, 1.06857992877536, 2.88689285692444, 1.43340446875028, 1.39014517551813, 1.42686683426878, 2.96969101563135, 0.688382491593920, 2.77215014120623, 0.0624404823330981, 1.07368604643331, 2.40654619584173, 1.07694959031671, 1.94403765328579, 1.42320795474861, 0.0319267577572077, 1.88206899462928, 1.88988270383512, 2.04020509305324,
1.07668862155395, 1.54974531733537, 2.20468787088922, 2.78911413048879, 0.172969325006747, 0.309013181227229, 2.04135282527034, 2.40039949872985, 3.10376455297212, 0.393719444023901, 1.14503881601765, 2.12443948376709, 1.18047911022321, 2.71263485347280, 0.917272015916727, 0.419324049857593, 2.11319391455861, 0.636439487634873, 2.72852096619963, 2.35983032417000, 1.31752002748297, 0.000725912732987877, 0.469554163584488, 0.860275830032366, 2.74080403038172, 1.88888617667581, 1.00904328383257, 0.893133034178407, 1.36758493744319, 2.83924279580859, 2.90630509808564, 1.58742305576421, 1.97160640152932,
2.25963426849743, 0.075124532405657, 1.80620372458346, 0.146192330442321, 1.32742150529536, 1.46943051910718, 0.0710879169578837, 0.204436133746323, 2.90269443655186, 1.67805929733473, 1.15232509236483, 1.14337064394727, 0.475554396507335, 0.470009274378247, 1.10207727395118, 1.05546861800760, 2.46309612204763, 1.52913633145838, 1.46020602264736, 0.412342051861392, 2.78468093314732, 2.11918465277978, 2.62373138338788, 2.06241767337499, 3.09106634510361, 3.07810066359534, 0.785878869792758, 1.96214673389555, 2.28784104661158, 1.56504820268029, 2.66981432825679, 0.599787225935080, 0.390009209925978,
0.00876636714795819, 0.480515361353021, 1.67811572253523, 1.60420278257283, 1.21019137827641, 0.975767537226525, 0.0111697675992302, 2.56115967709201, 2.00565073809155, 1.40849970472815, 0.766818062700615, 2.52390868417148, 2.58858088608758, 2.67723588360867, 1.46791468224683, 3.04954076297726, 2.64284601747040, 0.246766468035992, 0.746438920011734, 2.56847467438269, 1.27494850905113, 1.46496304663430, 2.98933919234972, 3.03165357951329, 2.40421331536636, 1.80495050613120, 2.87746333119346, 1.55644681580609, 0.521543264240427, 1.02415188507040, 0.931281492898919, 1.75394643950376, 0.211984628753030,
0.216700294916479, 0.523969218527704, 2.97646508560427, 2.54810941421016, 2.23196238576855, 3.04811632874419, 3.13665026654367, 3.10218076663472, 0.471510890906578, 3.01114997069145, 1.66648625210508, 0.232749658762213, 0.979614260977373, 2.81226194614244, 2.62250199511783, 0.00737883358857611, 2.01127854946753, 2.52327247738602, 0.770100979655590, 0.201445932247510, 0.826704826886323, 0.322705532317590, 1.51964813481983, 1.31596435056798, 1.19785474651322, 2.78585477078269, 1.32122039902513, 0.891734380041398, 0.151365528065576, 0.688530240709635, 0.751393608770545, 0.0919182479313140, 2.20637482165139,
0.0239878027790417, 1.91926392109170, 1.28205109286346, 0.782089044729246, 2.04976267613999, 1.00618014793620, 0.325695584698161, 1.68252673812153, 0.517955142561416, 2.77540453721136, 2.09373645298369, 2.66326158473405, 2.39597039496527, 2.53530854569149, 1.98847874705904, 2.23187321129420, 2.16350031528757, 1.00828744273377, 1.67022300461571, 2.74321972238607, 0.171341438884201, 1.57205540020913, 1.35956573550620, 2.84088723248477, 1.97977938725665, 3.08829937348963, 1.83846108752229, 2.64093750801654, 1.47282192897073, 1.71286646553928, 0.562672190577725, 1.99322020555813, 3.02522421440713]),

# Default static RIS
# Number of columns (x direction)
STATIC_RS1 = 32
# Number of rows (y direction)
STATIC_RS2 = 32

# Core electromagnetic far-field computation
def compute_fields(theta_inc_deg, phi_inc_deg,
                   theta_ref_deg, phi_ref_deg,
                   RS1_arg, RS2_arg,
                   DR1, DR2, randomize,
                   fc=28.5, ant_size_x=5.25, ant_size_y=5.25,
                   phi_ele_rand_sample_arg=None):
    """
    Core compute function moved from your script.
    Returns dict with masks, element coords, r/r_clip, and angular grids.
    """
    try:
        # Protect fc from zero / falsy values to avoid division by zero
        if not fc:
            fc = 28.5

        # Force static RIS size (override passed RS1_arg/RS2_arg)
        # RS1 = number of elements in x-direction
        RS1 = 32  # columns (x)
        # RS2 = number of elements in y-direction
        RS2 = 32  # rows    (y)

        # Convert angles from degrees to radians for trigonometric functions
        ti = np.deg2rad(theta_inc_deg)
        pi = np.deg2rad(phi_inc_deg)
        tr = np.deg2rad(theta_ref_deg)
        pr = np.deg2rad(phi_ref_deg)

        # Build incident unit direction vector in Cartesian coordinates
        incident_vector = np.array([np.sin(ti) * np.cos(pi),
                                    np.sin(ti) * np.sin(pi),
                                    np.cos(ti)])

        # Build reflected (desired beam) unit direction vector
        reflected_vector = np.array([np.sin(tr) * np.cos(pr),
                                     np.sin(tr) * np.sin(pr),
                                     np.cos(tr)])

        # Wavelength calculation: lambda = c / f
        # Here 300 corresponds to speed of light in mm/ns units
        lambda0 = 300.0 / fc

        # Wavenumber k = 2π / λ
        k0 = 2.0 * np.pi / lambda0

        print("\n=== Fundamental Parameters ===")
        print("Frequency (fc):", fc)
        print("Wavelength (lambda0):", lambda0)
        print("Wavenumber (k0):", k0)


        # Total physical size of RIS in x and y
        R_x = ant_size_x * RS1
        R_y = ant_size_y * RS2

        # Element center coordinates in x-direction
        x_vals = np.linspace(-R_x/2 + ant_size_x/2, R_x/2 - ant_size_x/2, RS1)

        # Element center coordinates in y-direction
        y_vals = np.linspace(-R_y/2 + ant_size_y/2, R_y/2 - ant_size_y/2, RS2)

        # Create 2D coordinate grid of element positions
        X, Y = np.meshgrid(x_vals, y_vals)   # shape (RS2, RS1)

        print("\n=== RIS Geometry ===")
        print("RIS size:", RS1, "x", RS2)
        print("Element spacing X:", ant_size_x)
        print("Element spacing Y:", ant_size_y)
        print("X shape:", X.shape)

        # Compute incident wavevector components
        kx_feed = k0 * np.sin(ti) * np.cos(pi)
        ky_feed = k0 * np.sin(ti) * np.sin(pi)

        # Compute incident phase at each RIS element
        kr = wrap_to_pi(kx_feed * X + ky_feed * Y)

        # Element randomization phase (per-element)
        if not randomize:
            # If randomization disabled, zero phase offset
            phi_ele_rand = np.zeros_like(X)
        else:
            # Total number of elements
            nels = RS1 * RS2
            # Flatten random sample
            sample = phi_ele_rand_sample_arg if phi_ele_rand_sample_arg is not None else phi_ele_rand_sample
            v = np.asarray(sample).copy().ravel()
            # If sample too small, tile it
            if v.size < nels:
                v = np.tile(v, int(np.ceil(nels / v.size)))[:nels]
            else:
                v = v[:nels]
            # Reshape to RIS grid
            phi_ele_rand = v.reshape((RS2, RS1))

        # Beam (reflected) wavevector components
        kx_beam = k0 * np.sin(tr) * np.cos(pr)
        ky_beam = k0 * np.sin(tr) * np.sin(pr)

        # Desired reflected phase at each element
        k_beam_r = wrap_to_pi(-(kx_beam * X + ky_beam * Y))

        # Required phase shift to steer beam
        phase_shift = wrap_to_pi(k_beam_r - kr - phi_ele_rand)

        # 1-bit quantization: 0 or π
        phase_shift_q = np.pi * (np.abs(phase_shift) >= (np.pi / 2))

        # Binary mask (0 or 1)
        mask = phase_shift_q / np.pi   # 0 or 1

        print("\n=== Phase Mask ===")
        print("Mask unique values:", np.unique(mask))
        print("Number of 0 states:", np.sum(mask == 0))
        print("Number of 1 states:", np.sum(mask == 1))

        # Observation grid angles
        theta = np.arange(-90, 91, 1)   # elevation angles
        phi = np.arange(0, 361, 1)      # azimuth angles

        # Convert observation grid to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        # Create angular meshgrid
        PHI, THETA = np.meshgrid(phi_rad, theta_rad, indexing='ij')

        # Direction cosines (u,v) representation
        U = np.sin(THETA) * np.cos(PHI)
        V = np.sin(THETA) * np.sin(PHI)

        # Static total phase at each element
        static_phase = kr + phase_shift_q + phi_ele_rand

        # Flatten element positions
        Xv = X.ravel()
        Yv = Y.ravel()

        # Flatten phase array
        base = static_phase.ravel()

        # Precompute kX and kY products
        kX = k0 * Xv
        kY = k0 * Yv

        # Flatten observation direction cosines
        Uv = U.ravel()
        Vv = V.ravel()

        # Use lower precision for memory/performance optimization
        dtype_f = np.float32
        dtype_c = np.complex64

        # Convert arrays to float32
        kX_f = kX.astype(dtype_f)
        kY_f = kY.astype(dtype_f)
        base_f = base.astype(dtype_f)
        Uv_f = Uv.astype(dtype_f)
        Vv_f = Vv.astype(dtype_f)

        # Number of elements
        nelem = base_f.size
        # Number of observation points
        nobs = Uv_f.size

        # Initialize far-field complex vector
        E_vec = np.zeros(nobs, dtype=dtype_c)

        # Chunk processing to avoid huge memory allocation
        chunk = 8192

        # Loop through observation points in chunks
        for start in range(0, nobs, chunk):
            end = min(start + chunk, nobs)
            Uc = Uv_f[start:end]
            Vc = Vv_f[start:end]

            # Compute phase for each element and observation
            phase_chunk = base_f[:, None] + (kX_f[:, None] * Uc[None, :]) + (kY_f[:, None] * Vc[None, :])

            # Sum complex exponentials across elements
            E_chunk = np.sum(np.exp(1j * phase_chunk).astype(dtype_c), axis=0)

            # Store results
            E_vec[start:end] = E_chunk

        # Reshape to angular grid
        E_rad = E_vec.reshape(U.shape)

        print("\n=== Field Computation ===")
        print("Number of elements:", nelem)
        print("Number of observation points:", nobs)
        print("E_rad shape:", E_rad.shape)

        # Avoid log(0)
        eps = 1e-12

        # Convert magnitude to dB
        r = 20.0 * np.log10(np.abs(E_rad) + eps)

        # Normalize so max is 0 dB
        r_norm = r - np.max(r)

        # Clip to dynamic range
        r_clip = np.clip(r_norm, DR1, DR2)

        print("\n=== Radiation Pattern ===")
        print("Max dB (should be 0):", np.max(r_norm))
        print("Min dB:", np.min(r_norm))
        print("Clipped range:", DR1, "to", DR2)

        # Return all relevant data
        return {
            "incident_vector": incident_vector,
            "reflected_vector": reflected_vector,
            "X": X, "Y": Y,
            "mask": mask,
            "r": r,
            "r_clip": r_clip,
            "x_vals": x_vals,
            "y_vals": y_vals,
            "theta": theta,
            "phi": phi
        }

    except Exception as e:
        # Capture full traceback on failure
        tb = traceback.format_exc()
        print("compute_fields error:", e)
        print(tb)
        return {"error": str(e), "trace": tb}


# Cache identical compute calls for speed
# compute_fields = lru_cache(maxsize=64)(compute_fields)

def build_vector_figure_visible(data, texture_rgb=None):
    """
    3D radiation-surface visualization:
      - constructs a parametric spherical surface where radius(θ,φ) ∝ 10^(dB/20)
      - maps clipped dB values (r_clip) to vertex intensity/color
      - overlays RIS plane grid and incident/reflected arrows (from data)
    """

    # Extract incident and reflected vectors from data dictionary
    # Convert to float numpy arrays to ensure numeric operations work properly
    iv = np.array(data["incident_vector"], dtype=float)
    rv = np.array(data["reflected_vector"], dtype=float)

    try:
        # Extract clipped dB radiation grid (phi x theta)
        r_clip = np.array(data["r_clip"])   # shape (len(phi), len(theta))

        # Extract angular coordinate arrays in degrees
        theta_deg = np.array(data["theta"]) # -90..90
        phi_deg = np.array(data["phi"])     # 0..360

        # Convert angular arrays to radians for trigonometric operations
        theta_rad = np.deg2rad(theta_deg)   # shape (len(theta),)
        phi_rad = np.deg2rad(phi_deg)       # shape (len(phi),)

        # Create 2D angular meshgrid (phi-major indexing)
        PH, TH = np.meshgrid(phi_rad, theta_rad, indexing='ij')  # shape (len(phi), len(theta))

        # Convert elevation angle (measured from XY-plane)
        # into co-latitude (measured from +Z axis)
        # Needed for proper spherical coordinate mapping
        co_lat = (np.pi / 2.0) - TH

        # Convert dB values to linear amplitude
        # amplitude = 10^(dB/20)
        amp = 10.0 ** (r_clip / 20.0)

        # Replace NaN or infinite values with 0
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)

        # Determine min/max amplitude for normalization
        a_min = amp.min() if amp.size else 0.0
        a_max = amp.max() if amp.size else 1.0

        # Prevent divide-by-zero during normalization
        if a_max - a_min < 1e-12:
            amp_norm = np.zeros_like(amp)
        else:
            # Normalize amplitude to range [0,1]
            amp_norm = (amp - a_min) / (a_max - a_min)

        # Base radius so even small sidelobes remain visible
        base_radius = 0.2

        # Overall scaling factor for visual size of radiation pattern
        scale_radius = 1.1

        # Final radius at each angular point
        R = base_radius + (amp_norm * scale_radius)

        # Convert spherical coordinates to Cartesian coordinates
        # X = R sin(θ) cos(φ)
        # Y = R sin(θ) sin(φ)
        # Z = R cos(θ)
        Xs = (R * np.sin(co_lat) * np.cos(PH))
        Ys = (R * np.sin(co_lat) * np.sin(PH))
        Zs = (R * np.cos(co_lat))

        # Get mesh dimensions
        rows, cols = Xs.shape   # rows=len(phi), cols=len(theta)

        # Flatten coordinate arrays into 1D vertex lists
        verts_x = Xs.ravel()
        verts_y = Ys.ravel()
        verts_z = Zs.ravel()

        # Use linear amplitude as color intensity for surface
        intensity = amp.ravel()

        # Initialize triangle index lists for Mesh3d
        I = []; J = []; K = []

        # Helper function converting 2D grid index to flattened index
        def idx(i, j): return i * cols + j

        # Build triangular faces (2 triangles per quad)
        for i in range(rows - 1):
            for j in range(cols - 1):

                # Get four corner indices of quad
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i + 1, j + 1)
                d = idx(i, j + 1)

                # First triangle (a-b-c)
                I.append(a); J.append(b); K.append(c)

                # Second triangle (a-c-d)
                I.append(a); J.append(c); K.append(d)

        # Create empty Plotly figure
        fig = go.Figure()

        # Add 3D radiation surface mesh
        fig.add_trace(go.Mesh3d(
            x=verts_x, y=verts_y, z=verts_z,   # vertex positions
            i=I, j=J, k=K,                     # triangle connectivity
            intensity=intensity,               # color mapping
            colorscale='Viridis',              # colormap
            showscale=True,                    # show colorbar
            colorbar=dict(title="Linear amplitude"),
            opacity=0.95,                      # slight transparency
            flatshading=False,                 # smooth shading
            name="Radiation surface",
            hoverinfo='skip'
        ))

        # Define half-width of RIS plane grid (visual reference)
        plane_half = 1.15

        # Determine grid resolution based on RIS element count
        ny = max(data["X"].shape[0], 8)
        nx = max(data["X"].shape[1], 8)

        # Create evenly spaced grid lines
        y_vals = np.linspace(-plane_half, plane_half, ny)
        z_vals = np.linspace(-plane_half, plane_half, nx)

        # Draw horizontal lines (parallel to z-axis)
        for yv in y_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[yv, yv], z=[z_vals[0], z_vals[-1]],
                mode='lines',
                line=dict(color='saddlebrown', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw vertical lines (parallel to y-axis)
        for zv in z_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[y_vals[0], y_vals[-1]], z=[zv, zv],
                mode='lines',
                line=dict(color='saddlebrown', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Helper function to draw a 3D arrow
        def add_arrow_line(fig, start, vec, color, name=None):

            # Ensure numeric numpy arrays
            start = np.array(start, dtype=float)
            vec = np.array(vec, dtype=float)

            # Compute vector magnitude
            norm = np.linalg.norm(vec)

            # If zero vector, replace with default Z-direction
            if norm < 1e-9:
                vec = np.array([0.,0.,1.])
                norm = 1.0

            # Normalize direction
            dir_u = vec / norm

            # Visible arrow length
            vis_len = 1.0

            # Compute arrow tip
            tip = start + dir_u * vis_len

            # Compute shortened line portion (for arrow shaft)
            line_end = start + dir_u * (vis_len * 0.86)

            # Draw arrow shaft
            fig.add_trace(go.Scatter3d(
                x=[start[0], line_end[0]],
                y=[start[1], line_end[1]],
                z=[start[2], line_end[2]],
                mode='lines',
                line=dict(color=color, width=6),
                name=name,
                hoverinfo='skip'
            ))

            # Draw arrow head marker
            fig.add_trace(go.Scatter3d(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw incident vector arrow (coming from +X direction)
        add_arrow_line(fig, [1.5, 0.0, 0.0], -iv, 'red', name='Incident Vector')

        # Draw reflected vector arrow (from origin outward)
        add_arrow_line(fig, [0.0, 0.0, 0.0], rv, 'blue', name='Reflected Vector')

        # Add origin marker
        fig.add_trace(go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode='markers',
            marker=dict(size=4, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Configure 3D scene layout
        fig.update_layout(
            title="3D Radiation Surface (mapped from r_clip)",
            scene=dict(
                xaxis=dict(title='X', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                yaxis=dict(title='Y', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                zaxis=dict(title='Z', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                aspectmode='auto',
                camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True
        )

        print("\n=== 3D Surface Stats ===")
        print("Amplitude min:", amp.min())
        print("Amplitude max:", amp.max())
        print("Vertices:", len(verts_x))

        # Return fully constructed figure
        return fig

    except Exception as e:
        # If error occurs, print message
        print("build_vector_figure_visible error:", e)

        # Create simple fallback 3D figure
        fallback = go.Figure()
        fallback.update_layout(scene=dict(aspectmode='cube'))

        # Return fallback figure
        return fallback
    
# Public service wrapper expected by tests / Dash
# This function acts as a lightweight interface layer between:
#   - Frontend (Dash app / API)
#   - Core electromagnetic computation (compute_fields)
def compute_farfield_pattern(params: dict):
    """
    Lightweight wrapper that computes a far-field pattern and returns a
    dictionary with keys `figure` (plotly Figure) and `pattern` (dB grid).
    Accepts a params dict with optional keys:
      - 'azi' : float (phi incidence in degrees)
      - 'dr_min', 'dr_max' : clipping range in dB
    """

    # Try to extract azimuth incidence angle from params dictionary
    # If missing or invalid, default to 0.0 degrees
    try:
        phi_inc = float(params.get("azi", 0.0))
    except Exception:
        # Fallback in case conversion fails
        phi_inc = 0.0

    # Extract dynamic range minimum (lower clipping bound in dB)
    # Default is -60 dB
    dr_min = float(params.get("dr_min", -60.0))

    # Extract dynamic range maximum (upper clipping bound in dB)
    # Default is 0 dB
    dr_max = float(params.get("dr_max", 0.0))

    # Call the core far-field computation function
    # Fixed configuration:
    #   - Incident elevation = 0°
    #   - Reflected direction = broadside (0°, 0°)
    #   - Static RIS size (32x32)
    #   - Randomization disabled
    data = compute_fields(
        theta_inc_deg=0.0,           # incident elevation
        phi_inc_deg=phi_inc,         # incident azimuth from params
        theta_ref_deg=0.0,           # reflection elevation
        phi_ref_deg=0.0,             # reflection azimuth
        RS1_arg=STATIC_RS1,          # unused (forced internally)
        RS2_arg=STATIC_RS2,          # unused (forced internally)
        DR1=dr_min,                  # lower dB clipping
        DR2=dr_max,                  # upper dB clipping
        randomize=False,             # no random element phase
    )

    # Build 3D radiation visualization from computed data
    fig = build_vector_figure_visible(data)

    # Extract clipped radiation pattern grid if computation succeeded
    # Otherwise return None
    pattern = data.get("r_clip") if isinstance(data, dict) else None

    # Return dictionary expected by frontend / tests
    #   "figure"  → interactive Plotly 3D radiation surface
    #   "pattern" → 2D numpy array of clipped dB values
    return {"figure": fig, "pattern": pattern}
def build_phase_figure(data):

    # Extract the 1-bit phase mask (values are 0 or 1)
    # Shape: (RS2 rows, RS1 columns)
    mask = np.array(data["mask"])   # shape (RS2, RS1)

    # Extract x-axis element coordinates (RIS horizontal positions)
    x = data["x_vals"]

    # Extract y-axis element coordinates (RIS vertical positions)
    y = data["y_vals"]

    # Convert binary mask (0 or 1) into phase degrees (0° or 180°)
    # Since 1-bit RIS uses only two phase states: 0 and π
    mask_deg = mask * 180.0

    # Flip vertically so orientation matches physical RIS layout
    # (Meshgrid origin vs plotting origin difference correction)
    mask_deg = np.flipud(mask_deg)

    # Create Plotly heatmap figure
    fig = go.Figure(
        go.Heatmap(

            # Z-values represent phase in degrees (0° or 180°)
            z=mask_deg,

            # X-axis positions (horizontal axis)
            x=x,

            # Y-axis positions (vertical axis)
            y=y,

            # Custom two-color scale:
            #   0°   → Gold
            #   180° → Indigo
            colorscale=[
                [0.0, "#FFD700"], [0.499, "#FFD700"],
                [0.5, "#4B0082"], [1.0, "#4B0082"]
            ],

            # Minimum phase value
            zmin=0,

            # Maximum phase value
            zmax=180,

            # Hide color bar (since only two discrete values)
            showscale=False
        )
    )

    # Configure layout properties
    fig.update_layout(

        # Title shown above heatmap
        title="Randomized Phase Mask",

        # X-axis label
        xaxis_title="y-axis (wavelength)",

        # Y-axis label
        yaxis_title="z-axis (wavelength)",

        # Force equal aspect ratio (square elements)
        yaxis_scaleanchor="x",

        # Margins around plot
        margin=dict(l=60, r=20, t=40, b=50)
    )

    # Return fully constructed phase heatmap figure
    return fig

def build_beam_figure(data, DR1, DR2, grid_size=480):

    # Extract clipped radiation pattern (dB values)
    # Shape: (len(phi), len(theta))
    r = np.array(data["r_clip"])        # shape (len(phi), len(theta))

    # Convert stored theta angles from degrees to radians
    theta = np.deg2rad(data["theta"])

    # Convert stored phi angles from degrees to radians
    phi = np.deg2rad(data["phi"])

    # Create angular meshgrid
    # TH and PH will match radiation grid dimensions
    TH, PH = np.meshgrid(theta, phi)   # TH,PH shape = (len(phi), len(theta))

    # Convert spherical angles to direction cosines
    # u = sin(theta) cos(phi)
    u = np.sin(TH) * np.cos(PH)

    # v = sin(theta) sin(phi)
    v = np.sin(TH) * np.sin(PH)

    # Flatten arrays for easier indexing
    u_flat = u.ravel()
    v_flat = v.ravel()
    r_flat = r.ravel()

    # Keep only points inside visible unit circle (u² + v² ≤ 1)
    # This corresponds to physically valid radiation directions
    inside = (u_flat**2 + v_flat**2) <= 1.0

    # Filter valid direction cosine points
    u_pts = u_flat[inside]
    v_pts = v_flat[inside]
    r_pts = r_flat[inside]

    # Create uniform grid for final 2D heatmap (v horizontal, u vertical)
    xs = np.linspace(-1.0, 1.0, grid_size)   # v axis (horizontal)
    ys = np.linspace(-1.0, 1.0, grid_size)   # u axis (vertical)

    # Initialize grid with NaN values
    Z = np.full((grid_size, grid_size), np.nan, dtype=float)

    # Map continuous v values into grid x-indices
    idx_x = np.clip(
        np.round(((v_pts + 1.0) / 2.0) * (grid_size - 1)).astype(int),
        0, grid_size-1
    )

    # Map continuous u values into grid y-indices
    idx_y = np.clip(
        np.round(((u_pts + 1.0) / 2.0) * (grid_size - 1)).astype(int),
        0, grid_size-1
    )

    # Populate grid with maximum dB value per pixel
    # If multiple angular points fall into same grid cell,
    # keep strongest value (better visual main lobe representation)
    for xi, yi, rv in zip(idx_x, idx_y, r_pts):
        cur = Z[yi, xi]
        if np.isnan(cur) or (rv > cur):
            Z[yi, xi] = rv

    # Fill small NaN holes by averaging neighbors (simple smoothing)
    for _ in range(2):

        # Identify NaN cells
        nan_mask = np.isnan(Z)

        # If no NaNs remain, stop early
        if not nan_mask.any():
            break

        # Shift grid in 4 directions to collect neighbors
        up = np.roll(Z, -1, axis=0)
        down = np.roll(Z, 1, axis=0)
        left = np.roll(Z, 1, axis=1)
        right = np.roll(Z, -1, axis=1)

        # Stack neighbors and compute sum ignoring NaNs
        neighbor_sum = np.nansum(
            np.stack([up, down, left, right], axis=0),
            axis=0
        )

        # Count valid neighbors per cell
        neighbor_count = np.sum(
            ~np.isnan(np.stack([up, down, left, right], axis=0)),
            axis=0
        )

        # Compute average of valid neighbors
        fill_vals = np.where(
            neighbor_count > 0,
            neighbor_sum / neighbor_count,
            np.nan
        )

        # Replace NaN cells with averaged values where possible
        Z = np.where(
            nan_mask & (~np.isnan(fill_vals)),
            fill_vals,
            Z
        )

    # Create meshgrid for circular mask
    XV, YU = np.meshgrid(xs, ys)

    # Define circular boundary (visible hemisphere)
    circle_mask = (XV**2 + YU**2) <= 1.0

    # Apply circular mask (outside circle → NaN)
    Z_masked = np.where(circle_mask, Z, np.nan)

    print("\n=== Beam Plot Stats ===")
    print("Grid size:", grid_size)
    print("Valid beam points:", np.sum(~np.isnan(Z_masked)))
    print("Max beam dB:", np.nanmax(Z_masked))
    print("Min beam dB:", np.nanmin(Z_masked))

    # Create 2D heatmap figure
    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=Z_masked,

            # Colormap for magnitude
            colorscale="Viridis",

            # Apply clipping range
            zmin=DR1,
            zmax=DR2,

            # Configure colorbar
            colorbar=dict(title="Magnitude dB", ticks="outside"),

            # Custom hover display
            hovertemplate="v=%{x:.3f}<br>u=%{y:.3f}<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    # Create circular outline (unit circle boundary)
    ang = np.linspace(0, 2*np.pi, 512)
    circ_x = np.cos(ang)
    circ_y = np.sin(ang)

    # Add circular boundary line
    fig.add_trace(
        go.Scatter(
            x=circ_x,
            y=circ_y,
            mode="lines",
            line=dict(color="black", width=1.8),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Configure layout of 2D beam plot
    fig.update_layout(
        title="Beam Pattern",
        xaxis=dict(title="v", range=[-1, 1], showgrid=False, zeroline=False),
        yaxis=dict(title="u", range=[-1, 1], scaleanchor="x", showgrid=False, zeroline=False),
        margin=dict(l=60, r=40, t=50, b=50),
        plot_bgcolor="white"
    )

    # Return fully constructed beam figure
    return fig