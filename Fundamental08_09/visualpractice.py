import matplotlib.pyplot as plt

# 그래프 데이터 
subject = ['English', 'Math', 'Korean', 'Science', 'Computer']
points = [40, 90, 50, 60, 100]

# 축 그리기
fig = plt.figure() #도화지(그래프) 객체 생성
ax1 = fig.add_subplot(1,1,1) #figure()객체에 add_subplot 메서드를 이용해 축을 그려준다.
plt.show()

fig = plt.figure(figsize=(5,2))
ax1 = fig.add_subplot(1,1,1)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,4)
plt.show()

# 축 그리기
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# 그래프 그리기
ax1.bar(subject,points)
plt.xlabel('Subject')
plt.ylabel('Points')
plt.title("Yuna's Test Result")
plt.show()