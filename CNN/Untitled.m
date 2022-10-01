a=0;
while a<=10
    fprintf('%d', a);
    a=a+1;
    
end


plot3(a(:,1),a(:,2),a(:,3),'.','color','black')
hold on
plot3(a(76,1),a(76,2),a(76,3),'x')
grid


X=[38 38 38 38];
Y=[max(a(:,2)) max(a(:,2)) min(a(:,2)) min(a(:,2))];
Z=[max(a(:,3)) min(a(:,3)) max(a(:,3)) min(a(:,3))];
vertex=[X;Y;Z]';
face=[1 2 4 3];

patch('Faces',face,'Vertices',vertex)
view(3)

ylabel('Maximum Stress (MPa)')
xlabel('R1')
zlabel('R2')


for i=1:100
    [MaxStress,R2_1,R2_2,result]=abaqus_result_analysis(a(i,1),a(i,2),a(i,3),a(i,4),a(i,5));
end

%37번 그림 실수로 지움