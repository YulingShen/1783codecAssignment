import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

def objective(x, a, b, c):
    return a * np.log(x) + b * x + c

def RD_plot(): 
    psnr = {'Default_test': [[40.029048516273406, 39.794388012160724, 39.66899819823547,
        39.608008467955294, 39.54296826209839, 39.493737537838726, 39.56743367835694,
        39.4592985330655, 40.00629262751075, 39.72768732723155],
        [35.953918715179356, 35.895205308252855, 35.74503366727245,
        35.62697616490776, 35.49659269056044, 35.31314084441265, 35.246657012845255,
        35.161129727084706, 35.967486513739495, 35.882453316615894],
        [31.170277680691093, 31.388582864193623, 31.375483194334137,
        31.39442723410805, 31.388567444518067, 31.414612070959617,
        31.378565270357065, 31.4323029873742, 31.017457099212216,
        31.270864254956386], [28.414493328086785, 28.493058048028516,
        28.55924578152588, 28.597688090738007, 28.612129561399296,
        28.643703571157143, 28.653574252816284, 28.68563710161538,
        28.072232417082393, 28.130451906764968]], 'MRF_test': [
        [40.029048516273406, 39.794388012160724, 39.71876129712766,
        39.617107631522906, 39.626630928627634, 39.57586850190468,
        39.597090558699385, 39.55950330482865, 40.00629262751075,
        39.72768732723155], [35.953918715179356, 35.895205308252855,
        35.788945295322506, 35.68296973974548, 35.61980697832916, 35.48541879090675,
        35.43499615022256, 35.36615020711521, 35.967486513739495,
        35.882453316615894], [31.170277680691093, 31.388582864193623,
        31.38896655963339, 31.40497224112229, 31.43472797865666, 31.446443996798816,
        31.480024339502457, 31.51071477124715, 31.017457099212216,
        31.270864254956386], [28.414493328086785, 28.493058048028516,
        28.560880618668854, 28.59983807999913, 28.624429050877588,
        28.660314794924517, 28.658570506372804, 28.693224834104107,
        28.072232417082393, 28.130451906764968]], 'VBSEnable_test': [
        [40.66457739624987, 40.54104466986701, 40.51173483409215, 40.48123017271205,
        40.5179803254429, 40.49748989571488, 40.50461762440507, 40.47652217363286,
        40.619947178757855, 40.52296387183401],[37.94138416945912, 37.33445961633318,
        37.00654544057266, 36.82241691664358, 36.69127915884089, 36.551084831121294,
        36.4614035658611, 36.37916004796847, 37.9914101143863, 37.32286687920941],
        [32.716244593834006, 32.76094525985998, 32.65109436924534, 32.59081490292449,
        32.56414586329377, 32.55442831086446, 32.57169546404384, 32.530125316842195,
        32.61708420484347, 32.736735050057206],
        [28.602367325681577, 28.696969015339192, 28.78947199230804,
        28.87740602344808, 28.87771484921871, 28.880678556117903, 28.916176856854637,
        28.931121615076623, 28.57390703397128,
        28.654050230129986]], 'FractionME_test': [
        [40.029048516273406, 40.58256882175242, 40.642537230609285,
        40.640559243835625, 40.65541998086616, 40.675170062332676, 40.60468459217595,
        40.776180665567765, 40.00629262751075, 40.51682994627191],
        [35.953918715179356, 36.3408627792472, 36.388084653078295, 36.42454311239804,
        36.3492711884165, 36.31968261044104, 36.342534077403684, 36.39839007772731,
        35.967486513739495, 36.25099326565476],
        [31.170277680691093, 31.56059743900248, 31.744373011886722,
        31.850569843667987, 31.92501153784758, 32.08270520853835, 32.14240779752996,
        32.25436616632151, 31.017457099212216, 31.517379001424292],
        [28.414493328086785, 28.548647756849522, 28.720126710862406,
        28.856186357583095, 29.00560477296367, 29.134855769004567,
        29.263017019112382, 29.420526757542248, 28.072232417082393,
        28.19583723689686]], 'FastME_test': [
        [40.029048516273406, 39.829326677746984, 39.70541886835288,
        39.63527854529549, 39.556171032149614, 39.51416636158797, 39.55509454053408,
        39.44331959585297, 40.00629262751075, 39.72861559913102],
        [35.953918715179356, 35.92440866302633, 35.79094948076106, 35.71175744337609,
        35.573566045948354, 35.406675944724554, 35.336792319656894,
        35.21888855037374, 35.967486513739495, 35.89156098333651],
        [31.170277680691093, 31.381739189979413, 31.42105175047229,
        31.377788891330695, 31.38296918679346, 31.37314618304522, 31.335960768929887,
        31.287181380552212, 31.017457099212216, 31.22987196389467],
        [28.414493328086785, 28.53837337961562, 28.611654327686203,
        28.633936083276268, 28.607761570948142, 28.635673391031986,
        28.622730298702038, 28.630338588067477, 28.072232417082393,
        28.15280875038166]], 'AllEnable_test': [
        [40.66457739624987, 40.98175933464215, 41.09305542002449, 41.1027092402005,
        41.146727809620494, 41.13969864156421, 41.146427551603864, 41.22732531459881,
        40.619947178757855, 40.96599614057001],[37.94138416945912, 38.07719602309707,
        38.10227612426202, 38.138332830259046, 38.141024391056234, 38.12806271567615,
        38.14181028080157, 38.130289559460415, 37.9914101143863,
        37.953223951653655], [32.716244593834006, 33.06966349102265,
        33.266773340071786, 33.39496345714247, 33.416014968379876, 33.44447874005386,
        33.52532256770834, 33.545274550461365, 32.61708420484347,
        32.92944778926981], [28.602367325681577, 28.790175016486902,
        28.981341374034542, 29.168738646111606, 29.31136307673703,
        29.444383224455702, 29.621481407089043, 29.852178923058684,
        28.57390703397128, 28.744105637922512]]}

    bits = {'Default_test': [[417524, 394312, 400762, 396247, 408727, 406834, 383711,
        418299, 421649, 391846], [128455, 94228, 95101, 90290, 98875, 100000, 86290,
        102145, 129642, 88784], [19606, 8184, 7579, 7270, 7562, 7033, 6108, 6493,
        20297, 8491], [2047, 3970, 4215, 4356, 4146, 4110, 4174, 4100, 1882,
        3975]], 'MRF_test': [[417524, 394312, 391053, 391805, 392170, 396384, 378399,
        397052, 421649, 391846], [128455, 94228, 91183, 88863, 90569, 93655, 84722,
        91002, 129642, 88784], [19606, 8184, 8572, 8620, 8456, 8360, 7344, 7822,
        20297, 8491], [2047, 3970, 5392, 5752, 5612, 5541, 5680, 5668, 1882,
        3975]], 'VBSEnable_test': [[503220, 542678, 550861, 552733, 560942, 560813,
        551888, 567791, 509580, 550311], [194727, 185949, 192876, 191454, 201687,
        201382, 191333, 206774, 196697, 185197], [40095, 28516, 29333, 26433, 26691,
        27564, 23362, 26358, 40653, 26051], [6272, 9198, 8789, 8729, 8950, 7981,
        8153, 7313, 6043, 8909]], 'FractionME_test': [[417524, 335645, 327013,
        322928, 333534, 328204, 318534, 327729, 421649, 341421], [128455, 75963,
        72755, 70575, 74178, 72992, 68595, 69663, 129642, 74417], [19606, 8485, 8064,
        7765, 7648, 6935, 6756, 6252, 20297, 8449], [2047, 4825, 5302, 5375, 5350,
        5484, 5519, 5537, 1882, 4904]], 'FastME_test': [[417524, 391556, 396347,
        392930, 405793, 405320, 382578, 418888, 421649, 395325], [128455, 91466,
        91952, 87146, 96951, 96979, 84370, 102314, 129642, 91317], [19606, 7782,
        7468, 6432, 7006, 6438, 5557, 5952, 20297, 8676], [2047, 3750, 3647, 3416,
        3270, 2937, 2883, 3057, 1882, 3608]], 'AllEnable_test': [[503220, 448671,
        440624, 444456, 452914, 461969, 441426, 436319, 509580, 465811],
        [194727, 170133, 164724, 168877, 171436, 174779, 157305, 165277, 196697,
        182826], [40095, 29419, 31529, 29254, 31651, 29509, 26476, 28255, 40653,
        29100], [6272, 9147, 14278, 16617, 16677, 16370, 15569, 15128, 6043, 8439]]}

    Default_test_1 =6.88
    Default_test_4 =6.09
    Default_test_7 =5.53
    Default_test_10= 5.55

    MRF_test_1 =9.57
    MRF_test_4 =8.16
    MRF_test_7 =7.69
    MRF_test_10= 7.88

    VBSEnable_test_1 =17.95
    VBSEnable_test_4 =15.37
    VBSEnable_test_7 =14.67
    VBSEnable_test_10= 14.31

    FractionME_test_1 =14.22
    FractionME_test_4 =13.3
    FractionME_test_7 =12.94
    FractionME_test_10= 13.03

    FastME_test_1 =5.97
    FastME_test_4 =4.9
    FastME_test_7 =4.45
    FastME_test_10= 4.49

    AllEnable_test_1 =20.9
    AllEnable_test_4 =18.61
    AllEnable_test_7 =18.71
    AllEnable_test_10= 15.31

    defaults = [round(Default_test_1, 2), round(Default_test_4, 2), round(Default_test_7, 2), round(Default_test_10, 2)]
    MRFs= [round(MRF_test_1, 2), round(MRF_test_4, 2), round(MRF_test_7, 2), round(MRF_test_10, 2)]
    VBSEnables = [round(VBSEnable_test_1, 2), round(VBSEnable_test_4, 2), round(VBSEnable_test_7, 2), round(VBSEnable_test_10, 2)]
    FMEEnables = [round(FractionME_test_1, 2), round(FractionME_test_4, 2), round(FractionME_test_7, 2), round(FractionME_test_10, 2)]
    FastMEEnables = [round(FastME_test_1, 2), round(FastME_test_4, 2), round(FastME_test_7, 2), round(FastME_test_10, 2)]
    AllEnables = [round(AllEnable_test_1, 2), round(AllEnable_test_4, 2), round(AllEnable_test_7, 2), round(AllEnable_test_10, 2)]

    mappers = {key:{"x":[],"y":[]} for key in psnr}

    plt.figure()
    plt.xlabel('Bit Size (Kbits)')
    plt.ylabel('PSNR (db)')
    colors = iter(cm.jet(np.linspace(0, 1, len(psnr))))
    for section in psnr:
        color = next(colors)
        print(color)
        print(section)
        # 1,4,7,10
        if section == "Default_test":
            execution_times = defaults
        elif section == "MRF_test":
            execution_times = MRFs
        elif section == "VBSEnable_only_test":
            execution_times = VBSEnables
        elif section == "FMEEnable_only_test":
            execution_times = FMEEnables
        elif section == "FastMEEnable_only_test":
            execution_times = FastMEEnables
        else:
            execution_times = AllEnables

        psnr_avg_1 = np.average(psnr[section][0])
        bits_sum_1 = np.average(bits[section][0])/1000
        psnr_avg_4 = np.average(psnr[section][1])
        bits_sum_4 = np.average(bits[section][1])/1000
        psnr_avg_7 = np.average(psnr[section][2])
        bits_sum_7 = np.average(bits[section][2])/1000
        psnr_avg_10 = np.average(psnr[section][3])
        bits_sum_10 = np.average(bits[section][3])/1000
        arr1_x, arr2_x, arr3_x, arr4_x = psnr_avg_1, psnr_avg_4, psnr_avg_7, psnr_avg_10
        arr1, arr2, arr3, arr4 = bits_sum_1, bits_sum_4, bits_sum_7, bits_sum_10
        y = [arr1_x, arr2_x, arr3_x, arr4_x]
        x = [arr1, arr2, arr3, arr4]

        popt, _ = curve_fit(objective, x, y)
        a, b, c= popt
        x_line = np.arange(min(x), max(x), 1)
        y_line = objective(x_line, a, b, c)

        print(x, y)
        plt.scatter(x, y, label="{}".format(section), marker='o', color=color)
        plt.plot(x_line, y_line, '--', c=color)
        
    plt.title("R-D Plots")    
    plt.legend()
    plt.show()
    # plt.savefig("RD Plots of different features.png")
    plt.close()

def Execution_time_plot():

    Default_test_1 =6.88
    Default_test_4 =6.09
    Default_test_7 =5.53
    Default_test_10= 5.55

    MRF_test_1 =9.57
    MRF_test_4 =8.16
    MRF_test_7 =7.69
    MRF_test_10= 7.88

    VBSEnable_test_1 =17.95
    VBSEnable_test_4 =15.37
    VBSEnable_test_7 =14.67
    VBSEnable_test_10= 14.31

    FractionME_test_1 =14.22
    FractionME_test_4 =13.3
    FractionME_test_7 =12.94
    FractionME_test_10= 13.03

    FastME_test_1 =5.97
    FastME_test_4 =4.9
    FastME_test_7 =4.45
    FastME_test_10= 4.49

    AllEnable_test_1 =20.9
    AllEnable_test_4 =18.61
    AllEnable_test_7 =18.71
    AllEnable_test_10= 15.31

    defaults = [round(Default_test_1, 2), round(Default_test_4, 2), round(Default_test_7, 2), round(Default_test_10, 2)]
    MRFs= [round(MRF_test_1, 2), round(MRF_test_4, 2), round(MRF_test_7, 2), round(MRF_test_10, 2)]
    VBSEnables = [round(VBSEnable_test_1, 2), round(VBSEnable_test_4, 2), round(VBSEnable_test_7, 2), round(VBSEnable_test_10, 2)]
    FMEEnables = [round(FractionME_test_1, 2), round(FractionME_test_4, 2), round(FractionME_test_7, 2), round(FractionME_test_10, 2)]
    FastMEEnables = [round(FastME_test_1, 2), round(FastME_test_4, 2), round(FastME_test_7, 2), round(FastME_test_10, 2)]
    AllEnables = [round(AllEnable_test_1, 2), round(AllEnable_test_4, 2), round(AllEnable_test_7, 2), round(AllEnable_test_10, 2)]

    width = 0.15  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    items = [defaults, MRFs, VBSEnables, FMEEnables, FastMEEnables, AllEnables]
    labels = ["qp=1", "qp=4", "qp=7", "qp=10"]
    species = ["Default", "Multiple Reference Frames", "Variable Block Size", "Fractional Motion Estimation", "Fast Motion Estimation", "All Features"]
    x = np.arange(len(labels))
    labels_loc = np.arange(6)
    for name, measurements in zip(species, items):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=name)
        ax.bar_label(rects, padding=6)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution time (s)')
    ax.set_title('Execution time of different settings')
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper left', ncols=6)
    ax.set_ylim(0, 30)
    ax.set_title('Total execution time of 10 frames in different settings')
    plt.show()
    # plt.savefig("time_plot.png")
    plt.close()

def Split_rate(): 
    split_rate = [[0.702020202020202, 0.851010101010101, 0.8813131313131313,
        0.9368686868686869, 0.9343434343434344, 0.9292929292929293,
        0.9545454545454546, 0.9696969696969697, 0.7272727272727273,
        0.8762626262626263], [0.9368686868686869, 0.8762626262626263,
        0.851010101010101, 0.8939393939393939, 0.8611111111111112,
        0.8939393939393939, 0.8358585858585859, 0.8964646464646465,
        0.9318181818181818, 0.8484848484848485],
        [0.8888888888888888, 0.5732323232323232, 0.6085858585858586,
        0.5606060606060606, 0.5984848484848485, 0.5606060606060606,
        0.4772727272727273, 0.4898989898989899, 0.8939393939393939,
        0.4444444444444444], [0.4823232323232323, 0.2601010101010101,
        0.2803030303030303, 0.3005050505050505, 0.3106060606060606,
        0.2878787878787879, 0.27525252525252525, 0.2676767676767677,
        0.44696969696969696, 0.24494949494949494]]

    bits = [[503220, 542678, 542576, 553227, 548616, 549481, 553235, 558554, 509580,
        550311], [194727, 185949, 182389, 190460, 183355, 193015, 182469, 192417,
        196697, 185197], [40095, 28516, 31630, 29688, 31281, 30007, 27994, 28169,
        40653, 26051], [6272, 9198, 11391, 12541, 13253, 12415, 12495, 12272, 6043,
        8909]]

    # labels = ["qp=1", "qp=4", "qp=7", "qp=10"]
    # width = 0.8  # the width of the bars
    # multiplier = 0
    # fig, ax = plt.subplots()
    # # species = ["Default", "Multiple Reference Frames", "Variable Block Size", "Fractional Motion Estimation", "Fast Motion Estimation", "All Features"]
    # x = np.arange(4)
    # labels_loc = np.arange(6)
    # for xs, name, sr in zip(x, labels, split_rate):
    #     # print(np.average(np.array(sr)*100))
    #     offset = width
    #     rects = ax.bar(xs+offset, np.average(np.array(sr)*100), width, label=name)
    #     ax.bar_label(rects)
    #     multiplier += 1
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Split Rate (%)')
    # ax.set_xticks(x + width, labels)
    # ax.legend(loc='upper left', ncols=1)
    # ax.set_ylim(0, 100)
    # ax.set_title('Split Rate with various QP values')
    # plt.show()
    # # plt.savefig("time_plot.png")
    # plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('QP values')
    ax.set_ylabel('Split Rate (%)')
    labels = ["qp=1", "qp=4", "qp=7", "qp=10"]
    # bit_labels = ["Size={}".format(np.average(bss)) for bss in bits]
    xs = [1,4,7,10]
    # xs = [np.average(bss) for bss in bits]
    # for x, sr, label in zip(xs, split_rate, labels):
    # print(xs, [np.average(np.array(sr)*100) for sr in split_rate])
    ax.plot(xs, [np.average(np.array(sr)*100) for sr in split_rate])
    for x, sr, label in zip(xs, split_rate, labels):
        avg_sr = round(np.average(np.array(sr)*100), 2)
        ax.scatter(x, avg_sr, label=label)
        ax.annotate(str(avg_sr)+"%", xy=(x-0.2, avg_sr+1))
    ax.set_title("Split Rate with various QP Values")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    # plt.xlabel('QP values')
    plt.xlabel('Bit Stream Size')
    plt.ylabel('Split Rate (%)')
    #　labels = ["qp=1", "qp=4", "qp=7", "qp=10"]
    bit_labels = ["Size={}".format(np.average(bss)) for bss in bits]
    # xs = [1,4,7,10]
    xs = [np.average(bss) for bss in bits]
    # for x, sr, label in zip(xs, split_rate, labels):
    # print(xs, [np.average(np.array(sr)*100) for sr in split_rate])
    plt.plot(xs, [np.average(np.array(sr)*100) for sr in split_rate])
    for x, sr, label in zip(bits, split_rate, bit_labels):
        plotx, ploty = np.average(x), round(np.average(np.array(sr)*100), 2)
        plt.scatter(plotx, ploty, label=label)
        ax.annotate(str(ploty)+"%", xy=(plotx-0.2, ploty+1))
    # plt.title("Split Rate with various Bit Stream Size")
    ax.set_title("Split Rate with various Bit Stream Size")
    plt.legend()
    plt.show()
    plt.close()

def test_nref():

    distortion = [[40.37050578121375, 40.25485642579543, 40.21776371044826, 35.72172881861121,
        35.69242309223934, 34.833378117336245, 34.85521485452304, 34.513383692647764,
        36.43331522647479, 35.635160268227445, 34.596975087063484,
        33.964173015791395, 33.96915062610802, 33.89639568399496, 33.53712090267021,
        33.1971011662013, 40.42463233856829, 35.76790230073708, 34.559321514143384,
        33.53071860876333, 33.41538426486726, 33.62816395536417, 33.38323011530332,
        33.111868858247085, 37.94138416945912, 36.4828307714326, 34.9613741692093,
        34.10444184403676, 33.32200360655543, 33.09112633072913],
        [40.37050578121375, 40.25485642579543, 40.28441852767994, 35.72489604115113,
        40.216212339630914, 35.465920622060295, 40.16669452511965, 35.46847494550937,
        36.43331522647479, 35.635160268227445, 34.57308069524766, 34.71167190269145,
        34.510426975212354, 34.02484673809537, 34.108099863541604,
        33.472889906949625, 40.42463233856829, 35.76790230073708, 35.694916170594155,
        34.041480780873826, 34.51389964872842, 34.012863079798194, 34.17278179045863,
        33.670504847634476, 37.94138416945912, 36.4828307714326, 35.26799991565452,
        34.77610271240272, 33.84176854484223, 33.81328988666987],
        [40.37050578121375, 40.25485642579543, 40.28441852767994, 35.74692613386182,
        40.24692710028664, 35.47476623725838, 40.16970781965653, 35.47290971836637,
        36.43331522647479, 35.635160268227445, 34.57308069524766, 35.50746991664654,
        36.34782745356106, 34.63242269651358, 35.090099699571844, 33.83144083158369,
        40.42463233856829, 35.76790230073708, 35.694916170594155, 34.135699398674404,
        34.96132461107139, 34.21059521690363, 34.206358725734695, 33.80110848680062,
        37.94138416945912, 36.4828307714326, 35.26799991565452, 35.09485414249838,
        33.975073379829944, 34.39047000254401],[40.37050578121375, 40.25485642579543,
        40.28441852767994, 35.74692613386182, 40.254828480221335, 35.47900272157069,
        40.228000016034485, 35.556086031801144, 36.43331522647479,
        35.635160268227445, 34.57308069524766, 35.50746991664654, 36.335226537471314,
        34.63479516795541, 35.0920740555516, 33.822789575902185, 40.42463233856829,
        35.76790230073708, 35.694916170594155, 34.135699398674404, 40.19948033585888,
        35.51695679327228, 35.35638062361342, 34.41965832880564, 37.94138416945912,
        36.4828307714326, 35.26799991565452, 35.09485414249838, 34.44098245569852,
        34.96063911484441]]

    bits = [[108843, 40679, 35363, 270221, 266662, 291779, 283072, 298375, 273083,
        301953, 297719, 354664, 321909, 311691, 363486, 424187, 108563, 270221,
        348918, 416352, 372182, 317191, 370222, 423479, 194727, 270495, 294384,
        353223, 418721, 389451], [108843, 40679, 40814, 272184, 30677, 216430, 30949,
        219837, 273083, 301953, 324145, 325580, 306865, 333023, 335629, 385558,
        108563, 270221, 313619, 369945, 343765, 322346, 335593, 379970, 194727,
        270495, 296029, 324893, 372419, 351953], [108843, 40679, 40814, 272086,
        36560, 217486, 34487, 222498, 273083, 301953, 324145, 288105, 141328, 238110,
        293432, 372679, 108563, 270221, 313619, 362804, 310584, 322577, 350584,
        372176, 194727, 270495, 296029, 312056, 365681, 325156], [108843, 40679,
        40814, 272086, 34864, 217467, 36617, 214446, 273083, 301953, 324145, 288105,
        140511, 238851, 293183, 374892, 108563, 270221, 313619, 362804, 38796,
        220253, 295340, 293772, 194727, 270495, 296029, 312056, 337633, 297571]]

    labels = ["nRefFrames=1", "nRefFrames=2", "nRefFrames=3", "nRefFrames=4"]
    # plt.figure()
    fig, ax = plt.subplots()
    # plt.xlabel('QP values')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Distortion in PSNR (db)')
    xs = list(range(1, 31))
    ys1 = [np.average(bss) for bss in bits]
    ys2 = [np.average(dis) for dis in distortion]
    for dis, label in zip(distortion, labels):
        ax.plot(xs, dis, label=label, marker="o")
    ax.set_title("Distortion For Different nRefFrames")
    plt.legend()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    # plt.xlabel('QP values')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Bits Stream Size')
    xs = list(range(1, 31))
    ys1 = [np.average(bss) for bss in bits]
    ys2 = [np.average(dis) for dis in distortion]
    # for x, sr, label in zip(xs, split_rate, labels):
    # print(xs, [np.average(np.array(sr)*100) for sr in split_rate])
    # plt.plot(xs, ys)
    for dis, label in zip(bits, labels):
        plt.plot(xs, dis, label=label, marker="o")
    ax.set_title("Bits Stream Size For Different nRefFrames")
    ax.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    RD_plot()
    # Execution_time_plot()
    # Split_rate()

    # test_nref()