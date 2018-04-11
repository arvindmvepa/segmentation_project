import xlsxwriter
import collections

f = '/Users/arvind.m.vepa/segmentation_project/out1.txt'

results = dict()
for line in open(f, 'r').readlines():
    line.replace(',', ' ')
    listOfWords = line.split(' ')

    if listOfWords[0] != "Step":
        continue
    step_num = listOfWords[1]
    metrics = []
    for i in range(9):
        i = i * 2 + 4
        metrics.append(listOfWords[i])
    if step_num in results:
        prev_results = results[step_num]
        for i in range(9):
            prev_results[i] = +metrics[i] / 3
            results[step_num] = prev_results
    else:
        results[step_num] = metrics

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()

row = 0
col = 0

for key in sorted(d.keys()):
    worksheet.write(row, col, key)
    for item in d[key]:
        worksheet.write(row, col, item)
        col += 1
    col = 0
    row += 1
workbook.close()
