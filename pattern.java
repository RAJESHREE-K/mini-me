import java.util.*;
class star1{
    public static void main(String []arg){
        Scanner sc=new Scanner(System.in);
        int n=sc.nextInt();
        for(int i=0;i<n;++i,System.out.println()){
            for(int j=0;j<=i;j++){
                System.out.print("*");
            }
        }
    }
}
