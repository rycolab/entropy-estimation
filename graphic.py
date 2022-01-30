import plotnine as p9
import pandas as pd
import matplotlib.pyplot as plt

data = {"10,2": {
"vals": [
["MLE", -0.04444322409942374, 0.11604042719855058, 0.026367094740885465],
["Horvitz-Thompson", -0.012289721092048681, 0.11241437574812305, 0.02522997724384085],
["Chao-Shen", 0.002158913777603281, 0.11570822007760741, 0.027213811447574635],
["Miller-Madow", -0.0029932240994236573, 0.1266728186163318, 0.028077725193687477],
["Jackknife", 0.011514175906576125, 0.12802795603382824, 0.02845696800044592],
["NSB", 0.14770071160140358, 0.19734516057978504, 0.051849647188431124]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0151, False, 0.0161, False],
["MLE", "Chao-Shen", 0.448, False, 0.8438, False],
["MLE", "Miller-Madow", 0.9999, True, 0.9999, True],
["MLE", "Jackknife", 0.9999, True, 0.9998, True],
["MLE", "NSB", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Chao-Shen", 0.9994, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Jackknife", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9079, False],
["Chao-Shen", "Jackknife", 0.9999, True, 0.9945, False],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.9726, False, 0.9629, False],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"100,2": {
"vals": [
["MLE", -0.002434475992036627, 0.0329941837764777, 0.0021382556265252986],
["Horvitz-Thompson", -0.0014775378285486838, 0.032799544804777925, 0.0021053146149388265],
["Chao-Shen", -0.0012661993716636642, 0.03289428899323013, 0.002108124494673359],
["Miller-Madow", 0.0024705240079633743, 0.03325888047091587, 0.0021496783446603575],
["Jackknife", 0.0027113137265169624, 0.03319615131945544, 0.002143999270657144],
["NSB", 0.013235317427186206, 0.03577339479546043, 0.0023062659627920896]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0394, False, 0.0017, True],
["MLE", "Chao-Shen", 0.2391, False, 0.0121, False],
["MLE", "Miller-Madow", 0.9584, False, 0.7822, False],
["MLE", "Jackknife", 0.8966, False, 0.6463, False],
["MLE", "NSB", 0.9999, True, 0.9997, True],
["Horvitz-Thompson", "Chao-Shen", 0.9849, False, 0.8016, False],
["Horvitz-Thompson", "Miller-Madow", 0.9968, True, 0.9983, True],
["Horvitz-Thompson", "Jackknife", 0.9913, False, 0.9959, False],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.9746, False, 0.9937, False],
["Chao-Shen", "Jackknife", 0.9489, False, 0.9847, False],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0043, False, 0.0011, True],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"1000,2": {
"vals": [
["MLE", -0.001288175550651364, 0.010504626094094327, 0.0002111288774958691],
["Horvitz-Thompson", -0.0012800338452331943, 0.010498195549599102, 0.00021109907535180598],
["Chao-Shen", -0.0012790393276921168, 0.010498850670444345, 0.00021109972688097812],
["Miller-Madow", -0.0007881755506514005, 0.010474643418325283, 0.00021009070194521746],
["Jackknife", -0.0007856309276595467, 0.010473697243679816, 0.0002100766041600453],
["NSB", 3.975505380896835e-05, 0.010469117226214292, 0.00020923266425605221]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0966, False, 0.1291, False],
["MLE", "Chao-Shen", 0.0949, False, 0.1278, False],
["MLE", "Miller-Madow", 0.0309, False, 0.0125, False],
["MLE", "Jackknife", 0.0251, False, 0.0115, False],
["MLE", "NSB", 0.1988, False, 0.0637, False],
["Horvitz-Thompson", "Chao-Shen", 0.9999, True, 0.5006, False],
["Horvitz-Thompson", "Miller-Madow", 0.0747, False, 0.015, False],
["Horvitz-Thompson", "Jackknife", 0.0659, False, 0.0143, False],
["Horvitz-Thompson", "NSB", 0.2393, False, 0.0671, False],
["Chao-Shen", "Miller-Madow", 0.0684, False, 0.0149, False],
["Chao-Shen", "Jackknife", 0.0554, False, 0.0132, False],
["Chao-Shen", "NSB", 0.2363, False, 0.0701, False],
["Miller-Madow", "Jackknife", 0.0046, False, 0.0013, True],
["Miller-Madow", "NSB", 0.4131, False, 0.129, False],
["Jackknife", "NSB", 0.4296, False, 0.138, False]]},
"10000,2": {
"vals": [
["MLE", -8.595578850650958e-05, 0.003296953668855467, 2.1094453724005197e-05],
["Horvitz-Thompson", -8.595578093123617e-05, 0.0032969536612801943, 2.109445370135357e-05],
["Chao-Shen", -8.595578093123617e-05, 0.0032969536612801943, 2.109445370135357e-05],
["Miller-Madow", -3.5955788506514955e-05, 0.003298585692266688, 2.108835814515456e-05],
["Jackknife", -3.593529267958054e-05, 0.00329858505806458, 2.1088352402136852e-05],
["NSB", 4.6336513501354954e-05, 0.003303200155022279, 2.108863893961411e-05]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0329, False, 0.1208, False],
["MLE", "Chao-Shen", 0.0353, False, 0.1271, False],
["MLE", "Miller-Madow", 0.8446, False, 0.3317, False],
["MLE", "Jackknife", 0.8514, False, 0.3407, False],
["MLE", "NSB", 0.9316, False, 0.4414, False],
["Horvitz-Thompson", "Chao-Shen", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.8509, False, 0.3399, False],
["Horvitz-Thompson", "Jackknife", 0.8541, False, 0.3344, False],
["Horvitz-Thompson", "NSB", 0.9313, False, 0.4439, False],
["Chao-Shen", "Miller-Madow", 0.8492, False, 0.3343, False],
["Chao-Shen", "Jackknife", 0.8497, False, 0.3432, False],
["Chao-Shen", "NSB", 0.9348, False, 0.4496, False],
["Miller-Madow", "Jackknife", 0.3857, False, 0.3145, False],
["Miller-Madow", "NSB", 0.964, False, 0.5116, False],
["Jackknife", "NSB", 0.9618, False, 0.5041, False]]},
"10,5": {
"vals": [
["MLE", -0.201762960798748, 0.23894411956462522, 0.09346309612978748],
["Horvitz-Thompson", -0.032371272953240555, 0.2288605409861487, 0.08524760407415244],
["Chao-Shen", -0.015151966014548277, 0.22653729156995786, 0.0841830922039591],
["Miller-Madow", -0.07426296079874815, 0.2095152320160739, 0.07396586500614288],
["Jackknife", -0.005785758722055018, 0.22426992005756838, 0.08071883244987177],
["NSB", 0.19027579683227283, 0.3040012634779115, 0.13797571769001007]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0265, False, 0.0023, True],
["MLE", "Chao-Shen", 0.0154, False, 0.0032, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0042, False, 0.0001, True],
["MLE", "NSB", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Chao-Shen", 0.0109, False, 0.0605, False],
["Horvitz-Thompson", "Miller-Madow", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Jackknife", 0.0006, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.0001, True, 0.0001, True],
["Chao-Shen", "Jackknife", 0.1209, False, 0.0018, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.9999, True, 0.9999, True],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"100,5": {
"vals": [
["MLE", -0.022120055549131826, 0.05533820985773583, 0.005049935806390004],
["Horvitz-Thompson", -0.013040348165926734, 0.05269941190286257, 0.004558663298664998],
["Chao-Shen", -0.013161202438341096, 0.052689790420863435, 0.004551894070367785],
["Miller-Madow", -0.0030900555491318333, 0.05261122058293761, 0.004625556590790262],
["Jackknife", -0.0012539302961726965, 0.052338543386898174, 0.0045746717153927],
["NSB", 0.009845634551868956, 0.05302395345681342, 0.004653491007265658]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0075, False, 0.0016, True],
["Horvitz-Thompson", "Chao-Shen", 0.4049, False, 0.0707, False],
["Horvitz-Thompson", "Miller-Madow", 0.4263, False, 0.8604, False],
["Horvitz-Thompson", "Jackknife", 0.2202, False, 0.6047, False],
["Horvitz-Thompson", "NSB", 0.6746, False, 0.8259, False],
["Chao-Shen", "Miller-Madow", 0.4415, False, 0.8777, False],
["Chao-Shen", "Jackknife", 0.2294, False, 0.645, False],
["Chao-Shen", "NSB", 0.6757, False, 0.8429, False],
["Miller-Madow", "Jackknife", 0.0006, True, 0.0002, True],
["Miller-Madow", "NSB", 0.844, False, 0.6765, False],
["Jackknife", "NSB", 0.977, False, 0.9404, False]]},
"1000,5": {
"vals": [
["MLE", -0.0016258288778063636, 0.01607620399634607, 0.0004403987678919504],
["Horvitz-Thompson", -0.0014543461079174148, 0.016032820791101247, 0.00043920266822054235],
["Chao-Shen", -0.0014570211875505243, 0.016034171610420003, 0.00043925762161930694],
["Miller-Madow", 0.00036217112219363834, 0.01598031481848099, 0.00043808365226783867],
["Jackknife", 0.0003908061780583699, 0.015975567094286717, 0.0004379582360079547],
["NSB", 0.001020028874111617, 0.015986659355040386, 0.0004384448201878431]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0222, False, 0.104, False],
["MLE", "Chao-Shen", 0.0234, False, 0.1117, False],
["MLE", "Miller-Madow", 0.0621, False, 0.1933, False],
["MLE", "Jackknife", 0.0572, False, 0.1823, False],
["MLE", "NSB", 0.1357, False, 0.2927, False],
["Horvitz-Thompson", "Chao-Shen", 0.9441, False, 0.9812, False],
["Horvitz-Thompson", "Miller-Madow", 0.1881, False, 0.3358, False],
["Horvitz-Thompson", "Jackknife", 0.1691, False, 0.3119, False],
["Horvitz-Thompson", "NSB", 0.274, False, 0.416, False],
["Chao-Shen", "Miller-Madow", 0.1877, False, 0.3302, False],
["Chao-Shen", "Jackknife", 0.1593, False, 0.3039, False],
["Chao-Shen", "NSB", 0.2759, False, 0.4108, False],
["Miller-Madow", "Jackknife", 0.0328, False, 0.1287, False],
["Miller-Madow", "NSB", 0.6185, False, 0.643, False],
["Jackknife", "NSB", 0.7029, False, 0.6927, False]]},
"10000,5": {
"vals": [
["MLE", -8.700908690301046e-06, 0.004983622307315511, 4.312236473289978e-05],
["Horvitz-Thompson", -6.7214589627632715e-06, 0.004982910045987336, 4.311895294090499e-05],
["Chao-Shen", -6.703702943998191e-06, 0.004982913246697747, 4.311916540015359e-05],
["Miller-Madow", 0.00019129909130967691, 0.0049890583535136266, 4.315888436942363e-05],
["Jackknife", 0.0001916587184047999, 0.004989044240882803, 4.315923611097639e-05],
["NSB", 0.00024792757378870394, 0.004992022265874877, 4.3191025037873264e-05]],
"comps": [
["MLE", "Horvitz-Thompson", 0.2259, False, 0.4461, False],
["MLE", "Chao-Shen", 0.2327, False, 0.4523, False],
["MLE", "Miller-Madow", 0.8065, False, 0.669, False],
["MLE", "Jackknife", 0.804, False, 0.6655, False],
["MLE", "NSB", 0.8534, False, 0.7346, False],
["Horvitz-Thompson", "Chao-Shen", 0.4999, False, 0.7479, False],
["Horvitz-Thompson", "Miller-Madow", 0.8291, False, 0.6766, False],
["Horvitz-Thompson", "Jackknife", 0.8377, False, 0.6804, False],
["Horvitz-Thompson", "NSB", 0.8736, False, 0.7469, False],
["Chao-Shen", "Miller-Madow", 0.8357, False, 0.6788, False],
["Chao-Shen", "Jackknife", 0.8297, False, 0.6775, False],
["Chao-Shen", "NSB", 0.8732, False, 0.748, False],
["Miller-Madow", "Jackknife", 0.3885, False, 0.5332, False],
["Miller-Madow", "NSB", 0.9503, False, 0.8939, False],
["Jackknife", "NSB", 0.9481, False, 0.8826, False]]},
"10,10": {
"vals": [
["MLE", -0.429158715292001, 0.43190088292193096, 0.2367325969541404],
["Horvitz-Thompson", -0.034859686368634124, 0.29458418846580914, 0.13504692878866173],
["Chao-Shen", -0.010587343085207167, 0.3062725824323592, 0.15034591746054748],
["Miller-Madow", -0.21365871529200123, 0.2758923270117497, 0.12150287551553961],
["Jackknife", -0.05758157154365851, 0.2645096324811759, 0.11093892466771293],
["NSB", 0.2951830075169701, 0.4268530132879235, 0.3042461095447725]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.3815, False, 0.9996, True],
["Horvitz-Thompson", "Chao-Shen", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.0014, True, 0.0007, True],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.0002, True, 0.0001, True],
["Chao-Shen", "Jackknife", 0.0001, True, 0.0001, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0102, False, 0.0003, True],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"100,10": {
"vals": [
["MLE", -0.04546143137930102, 0.06943652532325938, 0.007778698143515021],
["Horvitz-Thompson", -0.01134112119180428, 0.05890586338087166, 0.005690292500475771],
["Chao-Shen", -0.01711681889758421, 0.05908592839031419, 0.005756467121069213],
["Miller-Madow", -0.004381431379301041, 0.060504370851701114, 0.005944727906713418],
["Jackknife", 0.001994388628562803, 0.060131260866880844, 0.005857691203752327],
["NSB", 0.0170679206286205, 0.06214839485308426, 0.006161509469152631]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.7588, False, 0.9507, False],
["Horvitz-Thompson", "Miller-Madow", 0.9952, False, 0.9949, False],
["Horvitz-Thompson", "Jackknife", 0.9719, False, 0.9581, False],
["Horvitz-Thompson", "NSB", 0.9996, True, 0.9991, True],
["Chao-Shen", "Miller-Madow", 0.9886, False, 0.9769, False],
["Chao-Shen", "Jackknife", 0.9391, False, 0.8387, False],
["Chao-Shen", "NSB", 0.9983, True, 0.991, False],
["Miller-Madow", "Jackknife", 0.0533, False, 0.0104, False],
["Miller-Madow", "NSB", 0.9934, False, 0.9771, False],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"1000,10": {
"vals": [
["MLE", -0.004393548865461135, 0.018619946817379305, 0.0005729256685403385],
["Horvitz-Thompson", -0.003680759646005722, 0.01849379318559918, 0.0005649915605539621],
["Chao-Shen", -0.0037583507499917156, 0.01850324885663155, 0.0005655834169763631],
["Miller-Madow", 6.995113453888968e-05, 0.018305249989675067, 0.0005537708793066305],
["Jackknife", 0.0001796906089694743, 0.0182993587789654, 0.0005532760457563055],
["NSB", 0.0006558504495869264, 0.018288253188030117, 0.0005522640333711536]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0033, True, 0.0009, True],
["MLE", "Chao-Shen", 0.0019, True, 0.0006, True],
["MLE", "Miller-Madow", 0.0095, False, 0.002, True],
["MLE", "Jackknife", 0.0113, False, 0.0029, True],
["MLE", "NSB", 0.0152, False, 0.0038, False],
["Horvitz-Thompson", "Chao-Shen", 0.875, False, 0.9395, False],
["Horvitz-Thompson", "Miller-Madow", 0.0651, False, 0.028, False],
["Horvitz-Thompson", "Jackknife", 0.0535, False, 0.0257, False],
["Horvitz-Thompson", "NSB", 0.0648, False, 0.0293, False],
["Chao-Shen", "Miller-Madow", 0.0527, False, 0.0208, False],
["Chao-Shen", "Jackknife", 0.0489, False, 0.0185, False],
["Chao-Shen", "NSB", 0.0612, False, 0.0218, False],
["Miller-Madow", "Jackknife", 0.16, False, 0.0531, False],
["Miller-Madow", "NSB", 0.2006, False, 0.089, False],
["Jackknife", "NSB", 0.2627, False, 0.1404, False]]},
"10000,10": {
"vals": [
["MLE", -0.00041598127789435943, 0.005829269293771395, 5.456117402841645e-05],
["Horvitz-Thompson", -0.0004051147405865538, 0.0058290679560557826, 5.454005491937595e-05],
["Chao-Shen", -0.0004061026873404476, 0.005829033233993043, 5.453971913352183e-05],
["Miller-Madow", 3.3768722105621316e-05, 0.005820798009307596, 5.4390857375711584e-05],
["Jackknife", 3.534613438200052e-05, 0.005820755042164038, 5.4389306808385146e-05],
["NSB", 5.531343593982374e-05, 0.0058196045784315855, 5.4365104866794595e-05]],
"comps": [
["MLE", "Horvitz-Thompson", 0.4552, False, 0.2596, False],
["MLE", "Chao-Shen", 0.4506, False, 0.2291, False],
["MLE", "Miller-Madow", 0.271, False, 0.2084, False],
["MLE", "Jackknife", 0.2732, False, 0.2041, False],
["MLE", "NSB", 0.2579, False, 0.1937, False],
["Horvitz-Thompson", "Chao-Shen", 0.4538, False, 0.4793, False],
["Horvitz-Thompson", "Miller-Madow", 0.276, False, 0.2377, False],
["Horvitz-Thompson", "Jackknife", 0.2664, False, 0.2249, False],
["Horvitz-Thompson", "NSB", 0.2552, False, 0.2135, False],
["Chao-Shen", "Miller-Madow", 0.2866, False, 0.2425, False],
["Chao-Shen", "Jackknife", 0.2773, False, 0.2342, False],
["Chao-Shen", "NSB", 0.2516, False, 0.212, False],
["Miller-Madow", "Jackknife", 0.4152, False, 0.3037, False],
["Miller-Madow", "NSB", 0.1024, False, 0.0838, False],
["Jackknife", "NSB", 0.1027, False, 0.093, False]]},
"10,100": {
"vals": [
["MLE", -1.994717449492089, 1.994717449492089, 3.994489593639471],
["Horvitz-Thompson", -0.9263899706659632, 0.9263899706659632, 0.9378797811834919],
["Chao-Shen", -0.6559898009333112, 0.6559898009333112, 0.526270224159029],
["Miller-Madow", -1.5849174494920912, 1.5849174494920912, 2.538442160110814],
["Jackknife", -1.1562944240391133, 1.1562944240391133, 1.3895099547886025],
["NSB", -0.6915247797490549, 1.14101546271148, 1.7178457955227433]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Miller-Madow", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Jackknife", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9999, True],
["Chao-Shen", "Jackknife", 0.9999, True, 0.9999, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.0001, True, 0.0001, True],
["Jackknife", "NSB", 0.2827, False, 0.9999, True]]},
"100,100": {
"vals": [
["MLE", -0.4645192329223063, 0.4645192329223063, 0.22273499746819417],
["Horvitz-Thompson", 0.39266232621491387, 0.3946764880072264, 0.1850338326923341],
["Chao-Shen", -0.04967886662078064, 0.10504226153448029, 0.017906007876527394],
["Miller-Madow", -0.21877923292230575, 0.21955937285696095, 0.05742773218790777],
["Jackknife", -0.07291037212804027, 0.10551122875749457, 0.01794058089384647],
["NSB", 0.13390212608442212, 0.1614800048407998, 0.03828200405320902]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Miller-Madow", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.0001, True, 0.0001, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9999, True],
["Chao-Shen", "Jackknife", 0.6805, False, 0.5501, False],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.0001, True, 0.0001, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"1000,100": {
"vals": [
["MLE", -0.050207442732998396, 0.050703384678674246, 0.003258216531284713],
["Horvitz-Thompson", 0.007988314446658925, 0.023477930922790456, 0.0008630818504106438],
["Chao-Shen", -0.017325012400480358, 0.025974464496209233, 0.0010337189957877646],
["Miller-Madow", -0.005243942732998364, 0.022554976723703837, 0.0007919390835458317],
["Jackknife", 0.0015657777765768425, 0.022210772108521244, 0.0007654556600722085],
["NSB", 0.010471984566484717, 0.023770876018160355, 0.0008715770569488503]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.9996, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.0241, False, 0.009, False],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.86, False, 0.7051, False],
["Chao-Shen", "Miller-Madow", 0.0001, True, 0.0001, True],
["Chao-Shen", "Jackknife", 0.0001, True, 0.0001, True],
["Chao-Shen", "NSB", 0.0013, True, 0.0004, True],
["Miller-Madow", "Jackknife", 0.0539, False, 0.0146, False],
["Miller-Madow", "NSB", 0.9965, False, 0.9982, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"10000,100": {
"vals": [
["MLE", -0.004833161471065391, 0.0074838035779299875, 8.568345877983632e-05],
["Horvitz-Thompson", -0.0038270917917883956, 0.007095815888743569, 7.739260377593495e-05],
["Chao-Shen", -0.004133963693953966, 0.007195761270087751, 7.95386399378172e-05],
["Miller-Madow", 6.283852893472108e-05, 0.006259951287440206, 6.241110588980639e-05],
["Jackknife", 0.0001775705726577721, 0.0062610301177014865, 6.244619419928589e-05],
["NSB", 0.00014413076102385203, 0.006253488479188645, 6.229814500904202e-05]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.0001, True, 0.0001, True],
["Chao-Shen", "Miller-Madow", 0.0001, True, 0.0001, True],
["Chao-Shen", "Jackknife", 0.0001, True, 0.0001, True],
["Chao-Shen", "NSB", 0.0001, True, 0.0001, True],
["Miller-Madow", "Jackknife", 0.6094, False, 0.7067, False],
["Miller-Madow", "NSB", 0.0536, False, 0.0507, False],
["Jackknife", "NSB", 0.0081, False, 0.0008, True]]},
"10,1000": {
"vals": [
["MLE", -4.198367912555834, 4.198367912555834, 17.62886124538049],
["Horvitz-Thompson", -2.989072404494847, 2.989072404494847, 8.948748226676777],
["Chao-Shen", -2.927530181147656, 2.927530181147656, 8.579219479116865],
["Miller-Madow", -3.7540179125558333, 3.7540179125558333, 14.09713550714771],
["Jackknife", -3.265720438301342, 3.265720438301342, 10.674165044940144],
["NSB", -3.9317741632920704, 3.9317741632920704, 16.004966527178475]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Miller-Madow", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Jackknife", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9999, True],
["Chao-Shen", "Jackknife", 0.9999, True, 0.9999, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"100,1000": {
"vals": [
["MLE", -2.0101411035489747, 2.0101411035489747, 4.042812645535008],
["Horvitz-Thompson", 0.28339449366472685, 0.2914251491120457, 0.10582531192218007],
["Chao-Shen", -0.1511804868501707, 0.32376976325565493, 0.16174165088592174],
["Miller-Madow", -1.5603311035489784, 1.5603311035489784, 2.438247072380052],
["Jackknife", -1.1375032053266902, 1.1375032053266902, 1.3008545143364054],
["NSB", 0.455833241699781, 0.51203296275922, 0.41888133545211326]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.9991, True, 0.9999, True],
["Horvitz-Thompson", "Miller-Madow", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "Jackknife", 0.9999, True, 0.9999, True],
["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9999, True],
["Chao-Shen", "Jackknife", 0.9999, True, 0.9999, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.0001, True, 0.0001, True],
["Jackknife", "NSB", 0.0001, True, 0.0001, True]]},
"1000,1000": {
"vals": [
["MLE", -0.46773281708771963, 0.46773281708771963, 0.21947485816775417],
["Horvitz-Thompson", 0.8542175653540072, 0.8542175653540072, 0.7350455583400299],
["Chao-Shen", -0.057351360903909146, 0.061229364880316446, 0.0051077567002984796],
["Miller-Madow", -0.21786231708771994, 0.21786231708771994, 0.048428937831702185],
["Jackknife", -0.07177894547549817, 0.07231647202315167, 0.006428448090526497],
["NSB", 0.12728997948994858, 0.1273897213592394, 0.0182456073467035]],
"comps": [
["MLE", "Horvitz-Thompson", 0.9999, True, 0.9999, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Miller-Madow", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.0001, True, 0.0001, True],
["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9999, True],
["Chao-Shen", "Jackknife", 0.9999, True, 0.9999, True],
["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.0001, True, 0.0001, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]},
"10000,1000": {
"vals": [
["MLE", -0.05156726717034863, 0.05156726717034863, 0.002733333459690284],
["Horvitz-Thompson", 0.02757082703713103, 0.02757082703713103, 0.0008438517134540093],
["Chao-Shen", -0.016512119019913563, 0.016732247734220565, 0.0003458244992849723],
["Miller-Madow", -0.00614526717034862, 0.008631493391005279, 0.00011433988498766356],
["Jackknife", 0.000661805802163367, 0.006979029194429846, 7.655276953918308e-05],
["NSB", 0.009196966526610268, 0.010448204024324418, 0.0001602780473546845]],
"comps": [
["MLE", "Horvitz-Thompson", 0.0001, True, 0.0001, True],
["MLE", "Chao-Shen", 0.0001, True, 0.0001, True],
["MLE", "Miller-Madow", 0.0001, True, 0.0001, True],
["MLE", "Jackknife", 0.0001, True, 0.0001, True],
["MLE", "NSB", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Chao-Shen", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Miller-Madow", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "Jackknife", 0.0001, True, 0.0001, True],
["Horvitz-Thompson", "NSB", 0.0001, True, 0.0001, True],
["Chao-Shen", "Miller-Madow", 0.0001, True, 0.0001, True],
["Chao-Shen", "Jackknife", 0.0001, True, 0.0001, True],
["Chao-Shen", "NSB", 0.0001, True, 0.0001, True],
["Miller-Madow", "Jackknife", 0.0001, True, 0.0001, True],
["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
["Jackknife", "NSB", 0.9999, True, 0.9999, True]]}}

# ex = [
#     ["MLE", "Horvitz-Thompson", 0.0151, False, 0.0161, False],
#     ["MLE", "Chao-Shen", 0.448, False, 0.8438, False],
#     ["MLE", "Miller-Madow", 0.9999, True, 0.9999, True],
#     ["MLE", "Jackknife", 0.9999, True, 0.9998, True],
#     ["MLE", "NSB", 0.9999, True, 0.9999, True],
#     ["Horvitz-Thompson", "Chao-Shen", 0.9994, True, 0.9999, True],
#     ["Horvitz-Thompson", "Miller-Madow", 0.9999, True, 0.9999, True],
#     ["Horvitz-Thompson", "Jackknife", 0.9999, True, 0.9999, True],
#     ["Horvitz-Thompson", "NSB", 0.9999, True, 0.9999, True],
#     ["Chao-Shen", "Miller-Madow", 0.9999, True, 0.9079, False],
#     ["Chao-Shen", "Jackknife", 0.9999, True, 0.9945, False],
#     ["Chao-Shen", "NSB", 0.9999, True, 0.9999, True],
#     ["Miller-Madow", "Jackknife", 0.9726, False, 0.9629, False],
#     ["Miller-Madow", "NSB", 0.9999, True, 0.9999, True],
#     ["Jackknife", "NSB", 0.9999, True, 0.9999, True]
# ]
# more = []
# for x in ex:
#     more.append([x[1], x[0], 1 - x[2], x[3], 1 - x[4], x[5]])
# ex.extend(more)

shorten = {'Horvitz-Thompson': 'HT', 'Chao-Shen': 'CS', 'Miller-Madow': 'MM', 'Jackknife': 'J', 'MLE': 'MLE', 'NSB': 'NSB'} 

mod = []
for key in data:
    n, k = map(int, key.split(","))
    mab = min(data[key]["vals"], key=lambda x: x[2])[0]
    mse = min(data[key]["vals"], key=lambda x: x[3])[0]
    mab_true = sum([x[3] for x in data[key]["comps"] if mab in x[:2]])
    mse_true = sum([x[5] for x in data[key]["comps"] if mse in x[:2]])
    print(k, n, mab, mse, mab_true, mse_true)
    for row in data[key]["comps"]:
        mod.append([n, k, shorten[row[0]], shorten[row[1]], row[2], row[3], row[4], row[5]])
        mod.append([n, k, shorten[row[1]], shorten[row[0]], 1 - row[2], row[3], 1 - row[4], row[5]])

df = pd.DataFrame(mod, columns=['N', 'K', 'Estimator 1', 'Estimator 2', 'MAB', 'MAB-Sig', 'MSE', 'MSE-Sig'])
print(df.head())

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

labels = ['MLE', 'NSB', 'MM', 'J', 'HT', 'CS']

chart = (p9.ggplot(df, p9.aes('Estimator 1', 'Estimator 2', fill='MAB'))
    + p9.geom_tile(p9.aes(width=1, height=1))
    # + p9.scale_colour_grey(start=1, end=0)
    + p9.facet_grid('K ~ N', labeller='label_both')
    + p9.labs(title='Pairwise MAB p-values (Dirichlet)')
    + p9.scale_y_discrete(limits=labels)
    + p9.scale_x_discrete(limits=labels)
    + p9.theme(legend_key_width=2, legend_key_height=10, axis_text_x=p9.element_text(rotation=90, hjust=0.5))
    )
chart.draw()
chart.save('figures/mab.pdf', width=4, height=5)
plt.clf()

chart = (p9.ggplot(df, p9.aes('Estimator 1', 'Estimator 2', fill='MSE'))
    + p9.geom_tile(p9.aes(width=1, height=1))
    # + p9.scale_colour_grey(start=1, end=0)
    + p9.facet_grid('K ~ N', labeller='label_both')
    + p9.labs(title='Pairwise MSE p-values (Dirichlet)')
    + p9.scale_y_discrete(limits=labels)
    + p9.scale_x_discrete(limits=labels)
    + p9.theme(legend_key_width=2, legend_key_height=10, axis_text_x=p9.element_text(rotation=90, hjust=0.5))
    )
chart.draw()
chart.save('figures/mse.pdf', width=4, height=5)
plt.show()

