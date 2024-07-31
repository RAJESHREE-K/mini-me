import csv
f=open("student.csv","w",newline='')
writer=csv.writer(f)
print("enter student details")
ans='y'
while ans=='y':
    rno=int(input("enter roll no:"))
    name=input("enter name:")
    maths=int(input("enter maths mark:"))
    phy=int(input("enter physics mark:"))
    chem=int(input("enter chemistry mark:"))
    total=maths+phy+chem
    avg=total/3
    if maths>=40 and phy>=40 and chem>=40:
        result='pass'
    else:
        result='fail'
  
    writer.writerow([rno,name,maths,phy,chem,total,avg,result])
    ans=input("add more records y/n")
f.close()
print("student details")
f=open("student.csv","r")
reader=csv.reader(f)
for r in reader:
    print(r)
f.close()
