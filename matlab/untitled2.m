ue=[1 2 3 4 5 6];

tdoa=[40.7 41.5 41.8 42.1 42.5 43.3];
rgbd=[16.5 16.8 17 17.8 19.3 20.8];
two=[4 4.2 4.8 5.5 6 6.7];
three=[2.5 2.8 3.3 4 4.5 5];

%tdoa=[100 100 100 100 100 100];
%rgbd=[100 95 91.5 89 87.6 86.5];
%two=[100 96 93 91 90 89.5];
%three=[100 100 99.75 99.5 99 98.5];

plot(ue, tdoa, 'r--x')
hold on 
plot(ue, rgbd, 'k--o') 
plot(ue, two, 'm-*') 
plot(ue, three, 'b-diamond') 

xlabel('Number of UE(s)')
%ylabel('Distance error (cm)')
ylabel('Detection rate (%)')

xticks(1:6);
grid on
grid minor
hold off
legend("TDoA-based scheme", "RGB-d camera-based scheme", "C-IPM (two cameras)", "C-IPM (three cameras)")