#include <Windows.h>
#include <iostream>


void MoveCursorTo(int x, int y)
{
    SetCursorPos(x, y);
}

void MouseClick(int button)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = button;

    SendInput(1, &input, sizeof(INPUT));
}

int main()
{
    int startX = 2500; // ���� x ��ǥ
    int startY = 80; // ���� y ��ǥ
    int duration = 8; // ���α׷� ���� �ð� (��)
    int interval = 500; // ��ǥ ���� ���� (�и���)
    int step = -20; // x ��ǥ ���ҷ�

    int targetX = startX;
    int targetY = startY;

    // ���� �ð� ���� �ֱ������� x ��ǥ�� ���ҽ��� ���콺 Ŀ�� �̵� �� Ŭ��
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        targetX += step;

        MoveCursorTo(targetX, targetY);
        if (i%2 == 0)
            MouseClick(MOUSEEVENTF_RIGHTDOWN);
        else
            MouseClick(MOUSEEVENTF_RIGHTUP);
        Sleep(interval);
    }

    return 0;
}
