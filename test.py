import pandas as pd
import json

llp_processor_output = [{'pn': '335-009-306-0', 'sn': 'PA164966', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 0, 'csn': 6774, 
'tsn': None}, {'pn': '335-006-414-0', 'sn': 'DC055998', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 12722, 'csn': 19496, 'tsn': None}, 
{'pn': '1275M37P02', 'sn': 'GWN0804R', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 10536, 'csn': 17310, 'tsn': None}, 
{'pn': '1589M66G02', 'sn': 'GWNPV887', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 6871, 'operated_cycles_4': 0, 'csn': 13645, 'tsn': None}, 
{'pn': '1590M59P01', 'sn': 'XAEF2297', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 9842, 'csn': 16616, 'tsn': None}, 
{'pn': '1588M89G03', 'sn': 'GWN0917M', 'operated_cycles_1': 14146, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 0, 'csn': 14146, 'tsn': None}, 
{'pn': '1319M25P02', 'sn': 'GFF5EGET', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 0, 'csn': 6774, 'tsn': None}, 
{'pn': '1385M90P04', 'sn': 'GWNGK578', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 6871, 'operated_cycles_4': 0, 'csn': 13645, 'tsn': None}, 
{'pn': '1282M72P05', 'sn': 'XAEJ5247', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 0, 'csn': 6774, 'tsn': None},
 {'pn': '1475M29P02', 'sn': 'GWNGM885', 'operated_cycles_1': 6774, 'operated_cycles_2': 0, 'operated_cycles_3': 6165, 'operated_cycles_4': 0, 'csn': 12939, 'tsn': None}, 
 {'pn': '1864M91P02', 'sn': 'TMTTH701', 'operated_cycles_1': 14146, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 0, 'csn': 14146, 'tsn': None}, 
 {'pn': '301-331-126-0', 'sn': 'BB345134', 'operated_cycles_1': 19181, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 3427, 'csn': 22608, 'tsn': None}, 
 {'pn': '301-331-322-0', 'sn': '88370939', 'operated_cycles_1': 19181, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 3427, 
'csn': 22608, 'tsn': None}, {'pn': '305-056-116-0', 'sn': 'DB143578', 'operated_cycles_1': 19181, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 3427, 'csn': 22608, 'tsn': None}, 
{'pn': '301-330-626-0', 'sn': 'B8387368', 'operated_cycles_1': 19181, 'operated_cycles_2': 0, 'operated_cycles_3': 0, 'operated_cycles_4': 3427, 'csn': 22608, 'tsn': None}]

llp_processor_df = pd.DataFrame(llp_processor_output)
llp_processor_df.to_csv('llp_processor_output.csv')